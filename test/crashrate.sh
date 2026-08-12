#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
#
# Crash-rate harness: run a binary N times and report k/N crashed.
#
# The indexing crash is non-deterministic -- 5 of 8 runs on byte-identical
# input -- so a single trial proves nothing in either direction. Every
# experiment on this bug is a rate over N >= 12 trials. Decision rule:
# a variable "fixes" the crash only at 0/12 (confirm with 0/24 before
# believing it); it "doesn't matter" only if the rate stays within ~2
# crashes of the current baseline.
#
#   sh test/crashrate.sh [-n N] [-g] [-t SECS] -- <binary> [args...]
#
# -n N     trials (default 12)
# -g       after the trials, if anything crashed, re-run once under gdb
#          and capture bt / mappings / disassembly at rip. Cores from
#          apport are NOT trusted -- capture is always a live run with
#          "handle SIGSEGV stop nopass".
# -t SECS  per-trial timeout (default 300)
#
# Environment (SIGIL_OV_TOKENIZERS, SIGIL_OV_DEVICE, ...) passes through.

N=12
GDB=0
TMO=300

while [ $# -gt 0 ]; do
	case "$1" in
	-n) N="$2"; shift 2 ;;
	-g) GDB=1; shift ;;
	-t) TMO="$2"; shift 2 ;;
	--) shift; break ;;
	*)  break ;;
	esac
done

[ $# -ge 1 ] || { echo "usage: $0 [-n N] [-g] [-t SECS] -- binary [args...]" >&2; exit 2; }

BIN="$1"; shift
[ -x "$BIN" ] || { echo "not executable: $BIN" >&2; exit 2; }

ulimit -c unlimited 2>/dev/null

crashed=0
i=1
while [ "$i" -le "$N" ]; do
	t0=$(date +%s)
	timeout "$TMO" "$BIN" "$@" >/dev/null 2>&1
	rc=$?
	t1=$(date +%s)
	# 126/127 mean the binary could not be run at all (not executable, or
	# a failed link left it missing). Those are not passes, and reporting
	# them as 0/N crashed is exactly the false green this harness exists
	# to prevent -- one such run already claimed a fix that had not been
	# built. Abort rather than average it in.
	if [ "$rc" -eq 126 ] || [ "$rc" -eq 127 ]; then
		echo "  trial $i: rc=$rc -- cannot execute $BIN (build failed?)" >&2
		echo "ABORT: harness cannot run the binary; no rate reported" >&2
		exit 2
	fi
	# 124 = timeout (hang, counted separately as a crash: it is not a pass)
	# >= 128 = killed by signal (139 SIGSEGV, 134 SIGABRT, ...)
	if [ "$rc" -ge 128 ] || [ "$rc" -eq 124 ]; then
		crashed=$((crashed + 1))
		printf '  trial %2d/%d  rc=%-4s %3ds  CRASH\n' "$i" "$N" "$rc" "$((t1 - t0))"
	else
		printf '  trial %2d/%d  rc=%-4s %3ds\n' "$i" "$N" "$rc" "$((t1 - t0))"
	fi
	i=$((i + 1))
done

echo "== $crashed/$N crashed :: $BIN $*"

if [ "$GDB" -eq 1 ] && [ "$crashed" -gt 0 ]; then
	echo "== gdb capture (live, may take several attempts to hit the crash)"
	tries=0
	while [ "$tries" -lt 6 ]; do
		out=$(gdb -q -batch \
			-ex "handle SIGSEGV stop nopass" \
			-ex "run $*" \
			-ex "bt 15" \
			-ex "info registers rip rsp" \
			-ex 'printf "si_addr: %p\n", $_siginfo._sifields._sigfault.si_addr' \
			-ex 'x/16i $rip-0x20' \
			-ex "info proc mappings" \
			"$BIN" 2>&1)
		if echo "$out" | grep -q "SIGSEGV\|SIGABRT"; then
			echo "$out" | grep -vE '^\[New|^\[Thread'
			break
		fi
		tries=$((tries + 1))
		echo "  (clean run $tries, retrying for a crash)"
	done
fi

# Exit status: number of crashes, capped so it stays a valid exit code.
[ "$crashed" -gt 125 ] && crashed=125
exit "$crashed"
