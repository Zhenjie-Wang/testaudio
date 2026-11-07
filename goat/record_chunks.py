#!/usr/bin/env python3
"""
Record microphone audio in 0.5 s chunks, written to disk in real time so a
separate program can analyze them. On stop, all chunks are merged into one WAV
file and the temp chunk files are deleted.

Install:  pip install sounddevice numpy

------------------------------------------------------------------------------
Option 1: import it from another Python program
------------------------------------------------------------------------------
    from record_chunks import ChunkRecorder

    rec = ChunkRecorder(output="recording.wav", chunk_dir="chunks")
    rec.start()                      # returns immediately; records in background
    ...                              # do other work, analyze chunks, etc.
    path = rec.stop()                # merges chunks, deletes temp files

    # or as a context manager:
    with ChunkRecorder() as rec:
        time.sleep(10)

    # get a callback every time a chunk is finished:
    ChunkRecorder(on_chunk=lambda path: print("new chunk:", path))

------------------------------------------------------------------------------
Option 2: run it as a separate process and control it from anywhere
------------------------------------------------------------------------------
    python record_chunks.py --stop-file stop.flag

  Stop it any of these ways:
    * create the stop file (e.g. `touch stop.flag`)   <- works on every OS
    * send SIGINT or SIGTERM (Ctrl+C, `kill <pid>`, or Popen.terminate() on
      Linux/macOS; on Windows use CTRL_BREAK_EVENT or the stop file)
    * pass --duration N to stop automatically

  From Python:
    proc = subprocess.Popen([sys.executable, "record_chunks.py",
                             "--stop-file", "stop.flag"],
                            stdout=subprocess.PIPE, text=True)
    proc.stdout.readline()           # blocks until "RECORDING_STARTED"
    ...
    open("stop.flag", "w").close()   # or proc.terminate() / proc.send_signal(...)
    proc.wait()                      # final file is ready when this returns

Your analysis program should watch the chunk directory for new
`chunk_XXXXXX.wav` files. Chunks are written to a `.part` file first and then
renamed, so a finished `.wav` file is always complete and safe to read.
"""

import argparse
import glob
import os
import queue
import signal
import threading
import time
import wave

import numpy as np
import sounddevice as sd


class ChunkRecorder:
    def __init__(
        self,
        output="recording.wav",
        chunk_dir="chunks",
        sample_rate=44100,
        channels=1,
        device=None,
        chunk_seconds=0.5,
        on_chunk=None,
        remove_chunk_dir=False,
    ):
        self._output_arg = output
        self.output = self._resolve_output(output)
        self.chunk_dir = chunk_dir
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.chunk_seconds = chunk_seconds
        self.on_chunk = on_chunk
        self.remove_chunk_dir = remove_chunk_dir
        self.result = None  # path of the combined file after stop()

        self._done = threading.Event()
        self._queue = None
        self._stop_event = None
        self._writer = None
        self._stream = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ API
    @property
    def is_recording(self):
        return self._stream is not None

    def start(self):
        """Begin recording in the background and return immediately."""
        with self._lock:
            if self._stream is not None:
                raise RuntimeError("Already recording")

            self.output = self._resolve_output(self._output_arg)
            self.result = None
            self._done.clear()
            os.makedirs(self.chunk_dir, exist_ok=True)
            self._clear_stale_chunks()

            self._queue = queue.Queue()
            self._stop_event = threading.Event()
            self._writer = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer.start()

            def callback(indata, frames, time_info, status):
                if status:
                    print(f"Audio status: {status}", flush=True)
                self._queue.put(indata.copy())

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                device=self.device,
                callback=callback,
            )
            self._stream.start()

    def stop(self):
        """Stop recording, merge chunks into one file, delete temp chunks.

        Returns the path of the combined file (or None if nothing was recorded).
        Safe to call more than once.
        """
        with self._lock:
            if self._stream is None:
                return None
            self._stream.stop()
            self._stream.close()
            self._stream = None

            # Let the writer drain the queue and flush the final partial chunk
            self._stop_event.set()
            self._writer.join()

            try:
                self.result = self._combine_chunks()
            finally:
                self._done.set()
            return self.result

    def wait(self, timeout=None):
        """Block until recording has been stopped (by stop() from another thread,
        a callback, or a signal handler) and the combined file is written.
        Returns True if finished, False if `timeout` seconds passed first."""
        return self._done.wait(timeout)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # ------------------------------------------------------------ internals
    @staticmethod
    def _resolve_output(output):
        """If `output` is a folder (existing, or written with a trailing slash),
        put a timestamped .wav file inside it. Otherwise use it as the file path."""
        if os.path.isdir(output) or output.endswith(("/", os.sep)):
            os.makedirs(output, exist_ok=True)
            name = f"recording_{time.strftime('%Y%m%d_%H%M%S')}.wav"
            return os.path.join(output, name)
        return output

    def _chunk_path(self, index):
        return os.path.join(self.chunk_dir, f"chunk_{index:06d}.wav")

    def _clear_stale_chunks(self):
        for pattern in ("chunk_*.wav", "chunk_*.wav.part"):
            for f in glob.glob(os.path.join(self.chunk_dir, pattern)):
                os.remove(f)

    def _write_wav(self, path, samples):
        """Write atomically: .part file first, then rename."""
        tmp_path = path + ".part"
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(samples.tobytes())
        os.replace(tmp_path, path)  # readers never see a half-written file

    def _writer_loop(self):
        frames_per_chunk = int(self.sample_rate * self.chunk_seconds)
        buffer = np.empty((0, self.channels), dtype=np.int16)
        index = 0

        def flush(samples):
            nonlocal index
            path = self._chunk_path(index)
            self._write_wav(path, samples)
            index += 1
            if self.on_chunk:
                try:
                    self.on_chunk(path)
                except Exception as e:  # don't let a bad callback kill recording
                    print(f"on_chunk callback error: {e}", flush=True)

        while not (self._stop_event.is_set() and self._queue.empty()):
            try:
                block = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            buffer = np.concatenate([buffer, block])
            while len(buffer) >= frames_per_chunk:
                flush(buffer[:frames_per_chunk])
                buffer = buffer[frames_per_chunk:]

        if len(buffer) > 0:  # leftover audio shorter than one chunk
            flush(buffer)

    def _combine_chunks(self):
        files = sorted(glob.glob(os.path.join(self.chunk_dir, "chunk_*.wav")))
        if not files:
            return None

        parent = os.path.dirname(os.path.abspath(self.output))
        os.makedirs(parent, exist_ok=True)

        with wave.open(self.output, "wb") as out:
            out.setnchannels(self.channels)
            out.setsampwidth(2)
            out.setframerate(self.sample_rate)
            for f in files:
                with wave.open(f, "rb") as w:
                    out.writeframes(w.readframes(w.getnframes()))

        for f in files:
            os.remove(f)
        if self.remove_chunk_dir:
            try:
                os.rmdir(self.chunk_dir)  # only succeeds if empty
            except OSError:
                pass
        return self.output


# ---------------------------------------------------------------------- CLI
def main():
    parser = argparse.ArgumentParser(description="Record audio in 0.5 s chunks.")
    parser.add_argument("--output", default="recording.wav", help="final combined file")
    parser.add_argument("--chunk-dir", default="chunks", help="folder for temp chunks")
    parser.add_argument("--rate", type=int, default=44100, help="sample rate (Hz)")
    parser.add_argument("--channels", type=int, default=1, help="number of channels")
    parser.add_argument("--device", default=None, help="input device index or name")
    parser.add_argument("--duration", type=float, default=None,
                        help="stop automatically after this many seconds")
    parser.add_argument("--stop-file", default=None,
                        help="stop when this file appears (it is deleted on exit)")
    args = parser.parse_args()

    device = args.device
    if device is not None and device.isdigit():
        device = int(device)

    stop_requested = threading.Event()

    def handle_signal(signum, frame):
        stop_requested.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):  # SIGBREAK is Windows-only
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, handle_signal)

    # Remove a leftover stop file from a previous run so we don't quit instantly
    if args.stop_file and os.path.exists(args.stop_file):
        os.remove(args.stop_file)

    rec = ChunkRecorder(
        output=args.output,
        chunk_dir=args.chunk_dir,
        sample_rate=args.rate,
        channels=args.channels,
        device=device,
    )
    rec.start()
    # A controlling program can wait for this line before doing anything else
    print("RECORDING_STARTED", flush=True)

    start = time.time()
    while not stop_requested.is_set():
        if args.duration is not None and time.time() - start >= args.duration:
            break
        if args.stop_file and os.path.exists(args.stop_file):
            break
        time.sleep(0.05)

    path = rec.stop()
    if args.stop_file and os.path.exists(args.stop_file):
        os.remove(args.stop_file)

    if path:
        print(f"RECORDING_SAVED {path}", flush=True)
    else:
        print("RECORDING_EMPTY", flush=True)


if __name__ == "__main__":
    main()