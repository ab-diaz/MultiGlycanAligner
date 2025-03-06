import argparse
import subprocess
import sys
import shutil
import platform
import tarfile
import urllib.request
import itertools
import time
from pathlib import Path
from threading import Thread, Event
from queue import Queue

GSALIGN_URL = "https://glycanstructure.org/static/gsalign.tar.gz"
GSALIGN_ARCHIVE = "gsalign.tar.gz"
EXTRACT_DIR = Path("gsalign_src")
EXECUTABLE_NAME = "gsalign"


def animated_spinner(stop_event, message):
    """Show animated spinner while process is running."""
    spinner = ['-', '\\', '|', '/']
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{message} {spinner[i % 4]}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write("\r" + " " * 50 + "\r")  # Clear spinner line


def run_command_with_progress(cmd, cwd, timeout):
    """
    Run command with timeout and progress spinner.
    Returns a tuple:
      (status, stdout_lines, stderr_lines, elapsed_time, returncode)
    """
    start_time = time.time()
    stop_event = Event()

    proc = subprocess.Popen(cmd, cwd=cwd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)

    stdout_lines = []
    stderr_lines = []

    def read_stream(stream, collector):
        while True:
            line = stream.readline()
            if not line:
                break
            collector.append(line)
        stream.close()

    # Start threads to read stdout and stderr
    stdout_thread = Thread(target=read_stream, args=(proc.stdout, stdout_lines))
    stderr_thread = Thread(target=read_stream, args=(proc.stderr, stderr_lines))
    stdout_thread.start()
    stderr_thread.start()

    spinner = Thread(target=animated_spinner, args=(stop_event, "Processing..."))
    spinner.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stop_event.set()
        spinner.join()
        stdout_thread.join()
        stderr_thread.join()
        elapsed = time.time() - start_time
        return "timeout", stdout_lines, stderr_lines, elapsed, proc.returncode
    else:
        stop_event.set()
        spinner.join()
        stdout_thread.join()
        stderr_thread.join()
        elapsed = time.time() - start_time
        return "success", stdout_lines, stderr_lines, elapsed, proc.returncode


def check_gcc():
    """Check if g++ (GCC compiler) is installed, and install if missing (Linux only)."""
    if shutil.which("g++") is None:
        print("Error: g++ compiler not found. Required to compile GS-align.")
        if platform.system() == "Linux":
            install = input("Would you like to install g++ now? [y/N]: ").strip().lower()
            if install == "y":
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "g++"], check=True)
                print("g++ installed successfully.")
            else:
                print("g++ installation skipped. Compilation will fail.")
                sys.exit(1)
        else:
            print("Please install g++ manually before proceeding.")
            sys.exit(1)


def download_gsalign():
    """Download GS-align archive."""
    print(f"Downloading GS-align from {GSALIGN_URL}...")
    urllib.request.urlretrieve(GSALIGN_URL, GSALIGN_ARCHIVE)
    print(f"Downloaded GS-align archive to {GSALIGN_ARCHIVE}.")


def extract_gsalign():
    """Extract GS-align archive."""
    print(f"Extracting {GSALIGN_ARCHIVE}...")
    with tarfile.open(GSALIGN_ARCHIVE, "r:gz") as tar:
        tar.extractall(path=EXTRACT_DIR)
    print(f"Extracted GS-align to {EXTRACT_DIR}.")


def compile_gsalign() -> Path:
    """Compile the GS-align source code and return absolute path to executable."""
    source_dir = EXTRACT_DIR / "gsalign"
    source_file = "gsalign.cpp"
    output_executable = EXECUTABLE_NAME

    full_source_path = source_dir / source_file
    if not full_source_path.exists():
        print(f"Error: Source file {full_source_path} does not exist.")
        sys.exit(1)

    print("Compiling GS-align...")
    try:
        subprocess.run(
            ["g++", "-o", output_executable, source_file],
            check=True,
            cwd=str(source_dir),
        )
        compiled_path = (source_dir / output_executable).resolve()
        compiled_path.chmod(0o755)
        print(f"GS-align compiled successfully: {compiled_path}")
        return compiled_path
    except subprocess.CalledProcessError as e:
        print(f"Error compiling GS-align: {e}")
        sys.exit(1)


def check_gsalign(gsalign_path: str) -> Path:
    """Check if GS-align is available, and install if missing."""
    local_install_path = (EXTRACT_DIR / "gsalign" / EXECUTABLE_NAME).resolve()
    if local_install_path.exists():
        return local_install_path

    path_candidate = Path(gsalign_path).expanduser().resolve()
    if path_candidate.exists():
        return path_candidate

    if shutil.which(gsalign_path):
        return Path(shutil.which(gsalign_path)).resolve()

    print(f"GS-align not found at '{gsalign_path}' or in PATH.")
    install = input("Would you like to install GS-align now? [y/N]: ").strip().lower()
    if install == "y":
        download_gsalign()
        extract_gsalign()
        check_gcc()
        return compile_gsalign()

    print("GS-align installation skipped. Exiting.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Align glycan PDB files using GS-align.")
    parser.add_argument("-i", "--input_dir", required=True,
                        help="Input directory containing PDB files")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Output directory for aligned files")
    parser.add_argument("-g", "--gsalign_path", default=EXECUTABLE_NAME,
                        help="Path to GS-align executable")
    parser.add_argument("-t", "--timeout", type=int, default=1,
                        help="Maximum execution time per alignment in seconds (default: 1s)")

    args = parser.parse_args()

    gsalign_executable = check_gsalign(args.gsalign_path)
    print(f"Using GS-align at: {gsalign_executable}")

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(input_dir.glob("*.pdb"))
    if len(pdb_files) < 2:
        print("Error: At least two PDB files are required for alignment.")
        sys.exit(1)

    pairs = list(itertools.combinations(pdb_files, 2))
    total_pairs = len(pairs)
    print(f"\nFound {len(pdb_files)} PDB files. Will process {total_pairs} pairwise comparisons.")

    for i, (pdb1, pdb2) in enumerate(pairs, 1):
        pair_name = f"{pdb1.stem}_vs_{pdb2.stem}"
        pair_output = output_dir / pair_name
        pair_output.mkdir(exist_ok=True)

        # If running on Linux, prepend "stdbuf -oL" to force line buffering.
        if platform.system() == "Linux":
            cmd = [
                "stdbuf", "-oL", str(gsalign_executable),
                "-s1", str(pdb1.resolve()),
                "-s2", str(pdb2.resolve()),
                "-n", "1",
                "-o", "2",
            ]
        else:
            cmd = [
                str(gsalign_executable),
                "-s1", str(pdb1.resolve()),
                "-s2", str(pdb2.resolve()),
                "-n", "1",
                "-o", "2",
            ]

        print(f"\nProcessing pair {i}/{total_pairs}: {pdb1.name} vs {pdb2.name}")
        print(f"  Timeout set to: {args.timeout}s")

        result_type, stdout_lines, stderr_lines, elapsed, returncode = run_command_with_progress(
            cmd,
            cwd=str(pair_output),
            timeout=args.timeout
        )

        # Combine output lines into strings
        gs_output = "".join(stdout_lines).strip()
        err_output = "".join(stderr_lines).strip()
        # Save output to a file in the pair's folder
        output_file = pair_output / "gsalign_output.txt"
        with open(output_file, "w") as f:
            f.write(gs_output + "\n" + err_output)

        if result_type == "success" and returncode == 0:
            print(f"\r  Successfully processed {pair_name} in {elapsed:.1f}s")
            print("  GS-align output:")
            print(gs_output)
            if err_output:
                print("  GS-align error output:")
                print(err_output)
        elif result_type == "success" and returncode != 0:
            print(f"\r  ❌ Process finished with errors (return code {returncode}) after {elapsed:.1f}s")
            print("  GS-align output:")
            print(gs_output)
            if err_output:
                print("  GS-align error output:")
                print(err_output)
        elif result_type == "timeout":
            print(f"\r  ⚠️ Timed out after {elapsed:.1f}s")
            print("  Partial GS-align output:")
            print(gs_output)

    print(f"\n✅ Completed all {total_pairs} pairwise comparisons.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
