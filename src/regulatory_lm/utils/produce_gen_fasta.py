#!/usr/bin/env python3
import argparse
import sys
from pyfaidx import Fasta

WINDOW = 2114  # output sequence length


def read_sequences(path):
    seqs = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            s = s.upper()
            bad = set(s) - set("ACGTN")
            if bad:
                raise ValueError(f"Line {i}: invalid characters {sorted(bad)} in sequence: {s[:50]}...")
            seqs.append(s)
    if not seqs:
        raise ValueError("No sequences found in input text file.")
    return seqs


def wrap_fasta(seq, width=60):
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-seqs", required=True, help="Text file: one DNA sequence per line")
    ap.add_argument("-g", "--genome", required=True, help="Reference genome FASTA")
    ap.add_argument("-c", "--chrom", required=True, help="Chromosome name (must match FASTA)")
    ap.add_argument("-s", "--start", required=True, type=int, help="Start coordinate (0-based, BED-style)")
    ap.add_argument("-e", "--end", required=True, type=int, help="End coordinate (0-based, end-exclusive)")
    ap.add_argument("-o", "--out-prefix", required=True, help="Output prefix (writes .fa and .narrowPeak)")
    args = ap.parse_args()

    inserts = read_sequences(args.input_seqs)

    fa = Fasta(args.genome, as_raw=True, sequence_always_upper=True)
    if args.chrom not in fa.keys():
        raise ValueError(f"Chromosome '{args.chrom}' not found in FASTA.")
    if args.end <= args.start:
        raise ValueError("end must be > start")

    # 2,114bp window centered on midpoint (0-based, end-exclusive)
    midpoint = (args.start + args.end) // 2
    win_start = midpoint - (WINDOW // 2)
    win_end = midpoint + (WINDOW // 2)
    if (win_end - win_start) != WINDOW:
        raise RuntimeError("Window size mismatch; check WINDOW parity/logic.")

    chr_len = len(fa[args.chrom])
    if win_start < 0 or win_end > chr_len:
        raise ValueError(f"Requested window [{win_start}, {win_end}) out of bounds for {args.chrom} length {chr_len}.")

    ref_window = str(fa[args.chrom][win_start:win_end])
    if len(ref_window) != WINDOW:
        raise RuntimeError("Fetched window length mismatch; check FASTA/coordinates.")

    center = WINDOW // 2  # 1057

    out_fa_path = args.out_prefix + ".fa"
    out_np_path = args.out_prefix + ".narrowPeak"

    with open(out_fa_path, "w") as fa_out, open(out_np_path, "w") as np_out:
        for idx, ins in enumerate(inserts, 1):
            name = f"seq_{idx}"
            L = len(ins)

            # Replace centered interval [rep_start, rep_end) in the 2114bp window
            rep_start = center - (L // 2)
            rep_end = rep_start + L
            if rep_start < 0 or rep_end > WINDOW:
                raise ValueError(f"{name}: insert length {L} too long to fit centered within {WINDOW}bp.")

            new_seq = ref_window[:rep_start] + ins + ref_window[rep_end:]

            fa_out.write(f">{name}\n")
            fa_out.write(wrap_fasta(new_seq) + "\n")

            # narrowPeak (10 columns):
            # chrom, start, end reflect the generated FASTA record coordinate system
            # other columns can be dummy; 10th column (peak) must be 1057
            # Columns: chrom start end name score strand signalValue pValue qValue peak
            np_out.write(
                f"{name}\t0\t{WINDOW}\t{name}\t0\t.\t0\t-1\t-1\t{center}\n"
            )

    fa.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)







