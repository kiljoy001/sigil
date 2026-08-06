/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Balanced-ternary packing for the two trailing sigil bytes.
 *
 * Six trits, base-3, most-significant trit first. Every packed value below
 * 3^6 = 729 decodes to a legal tuple, so validity is a single comparison and
 * corruption is detectable rather than silent.
 */

#include "sigil.h"

/* Balanced {-1,0,+1} maps to unbalanced {0,1,2} for packing. */
static inline unsigned trit_to_digit(trit_t t)
{
	return (unsigned)((int)t + 1);
}

static inline trit_t digit_to_trit(unsigned d)
{
	return (trit_t)((int)d - 1);
}

uint16_t sigil_trits_pack(const sigil_trits_t *t)
{
	const trit_t order[SIGIL_NTRITS] = {
		t->thermal, t->quality,
		t->sim[0], t->sim[1], t->sim[2], t->sim[3]
	};
	unsigned acc = 0;

	for (int i = 0; i < SIGIL_NTRITS; i++)
		acc = acc * 3 + trit_to_digit(order[i]);

	return (uint16_t)acc;
}

int sigil_trits_unpack(uint16_t packed, sigil_trits_t *out)
{
	trit_t order[SIGIL_NTRITS];
	unsigned acc = packed;

	if (!sigil_trits_valid(packed))
		return -1;

	/* Least-significant trit falls out first; fill backwards. */
	for (int i = SIGIL_NTRITS - 1; i >= 0; i--) {
		order[i] = digit_to_trit(acc % 3);
		acc /= 3;
	}

	out->thermal = order[0];
	out->quality = order[1];
	out->sim[0]  = order[2];
	out->sim[1]  = order[3];
	out->sim[2]  = order[4];
	out->sim[3]  = order[5];

	return 0;
}
