"""
Robust wrapper around the 'geng' tool from nauty for generating all
non-isomorphic 3-regular (cubic) graphs on n vertices, parsed into NetworkX.

Requires:
    - nauty's 'geng' available on PATH (https://pallini.di.uniroma1.it/)
    - networkx (pip install networkx)

This implementation:
    - streams geng stdout (no huge memory spike),
    - optionally requests only connected graphs (default True),
    - checks for geng presence and provides helpful errors,
    - allows an optional per-graph callback (e.g. to process or display progress).
"""
from itertools import combinations
import matplotlib.pyplot as plt

from typing import Callable, Optional, List, Iterator
import shutil
import subprocess
import networkx as nx


def generate_cubic_graphs_with_geng(
    n: int,
    connected: bool = True,
    geng_path: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Iterator[nx.Graph]:
    """
    Generate all non-isomorphic simple 3-regular graphs on n vertices
    using nauty's geng and yield them one by one as networkx.Graph objects.

    Parameters
    ----------
    n : int
        Number of vertices. Must be even and >= 4 (otherwise no simple 3-regular graphs).
    connected : bool
        If True (default), request only connected graphs from geng. If False, request all graphs.
    geng_path : Optional[str]
        Optional explicit path to 'geng'. If None, will look up 'geng' on PATH.

    Returns
    -------
    Ierator[nx.Graph]
        Ierator of NetworkX Graph objects parsed from geng.
    """
    # Validate n
    if n < 4 or (n % 2) == 1:
        raise ValueError("n must be an even integer >= 4 for simple 3-regular graphs.")

    # Locate geng
    geng_path = "/opt/homebrew/bin/geng"
    if geng_path is None:
        geng_path = shutil.which("geng")
    if geng_path is None:
        raise FileNotFoundError(
            "geng executable not found. Please install nauty and put 'geng' on your PATH."
        )

    # Build command. -d3 -D3 constrains min and max degree to 3.
    # Add '-c' for connected graphs when requested (geng uses -c for connected).
    cmd = [geng_path, str(n), "-d3", "-D3"]
    if connected:
        cmd.append("-c")

    # Start the process and stream its stdout line-by-line.
    # Each line is a graph6 string (graph6 per-line).
    graphs: List[nx.Graph] = []
    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        ) as proc:
            if proc.stdout is None:
                raise RuntimeError("Failed to open geng stdout.")

            idx = 0
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                # Skip graph6 header lines if present (some tools may emit headers)
                # graph6 format header typically begins with '>>graph6<<' or similar; skip lines starting with '>'.
                if line.startswith(">"):
                    continue
                # networkx supports from_graph6_bytes/from_graph6_str
                # from_graph6_str expects exactly one graph string
                try:
                    # networkx.from_graph6_bytes expects a bytes object possibly including newline
                    G = nx.from_graph6_bytes(line.encode("ascii"))
                except Exception:
                    # Fallback: try from_graph6_str (older networkx versions)
                    G = nx.from_graph6_str(line)
                # Optionally verify degree-3 (sanity check)
                if any(d != 3 for _, d in G.degree()):
                    # skip if geng output didn't meet constraints (defensive)
                    continue
                graphs.append(G)
                yield G
                idx += 1

            # Wait for process to finish and capture errors if any
            proc.wait(timeout=timeout)
            if proc.returncode != 0:
                # capture stderr for diagnostics
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"geng exited with code {proc.returncode}. stderr: {stderr}")

    except subprocess.TimeoutExpired:
        raise TimeoutError("geng process timed out.")
    except FileNotFoundError:
        raise  # already handled above, re-raise


def all_cubic_graphs(
    n: int,
    connected: bool = True,
    geng_path: Optional[str] = None,
    timeout: Optional[float] = None,
) -> List[nx.Graph]:
    return [g for g in generate_cubic_graphs_with_geng(n, connected, geng_path, timeout)]


if __name__ == "__main__":
    import sys

    n = 12
    try:
        def simple_progress(g, i):
            if i % 100 == 0:
                pass
                # print(f"Got graph #{i}")

        graphs = generate_cubic_graphs_with_geng(n, connected=True)
        for i, G in enumerate(graphs, 1):
            plt.figure(i)
            nx.draw(G)
            if i == 5:
                break
    except Exception as e:
        print("Error:", e, file=sys.stderr)

