#ifndef _MADNG_PANIC_H
#define _MADNG_PANIC_H

#include "mad_tpsa.h"
#include "mad_ctpsa.h"
#include "madng_log.h"


/* ------------------------------------------------------------------------- */
/* Function signatures                                                       */
/* ------------------------------------------------------------------------- */

typedef void (*madng_tpsa_unary_fn)(const tpsa_t *input, tpsa_t *output);
typedef void (*madng_tpsa_binary_fn)(const tpsa_t *left, const tpsa_t *right, tpsa_t *output);
typedef void (*madng_tpsa_two_output_fn)(
    const tpsa_t *input,
    tpsa_t *first_output,
    tpsa_t *second_output
);

/* compose, exppb */
typedef void (*madng_tpsa_map_binary_fn)(
    ssz_t na,
    const tpsa_t *ma[],
    ssz_t nb,
    const tpsa_t *mb[],
    tpsa_t *mc[]
);
typedef void (*madng_ctpsa_map_binary_fn)(
    ssz_t na,
    const ctpsa_t *ma[],
    ssz_t nb,
    const ctpsa_t *mb[],
    ctpsa_t *mc[]
);

/* liebra, logpb */
typedef void (*madng_tpsa_map_pair_fn)(
    ssz_t na,
    const tpsa_t *ma[],
    const tpsa_t *mb[],
    tpsa_t *mc[]
);
typedef void (*madng_ctpsa_map_pair_fn)(
    ssz_t na,
    const ctpsa_t *ma[],
    const ctpsa_t *mb[],
    ctpsa_t *mc[]
);

/* minv */
typedef void (*madng_tpsa_map_inverse_fn)(
    ssz_t na,
    const tpsa_t *ma[],
    ssz_t nb,
    tpsa_t *mc[]
);
typedef void (*madng_ctpsa_map_inverse_fn)(
    ssz_t na,
    const ctpsa_t *ma[],
    ssz_t nb,
    ctpsa_t *mc[]
);

/* pminv */
typedef void (*madng_tpsa_map_partial_inverse_fn)(
    ssz_t na,
    const tpsa_t *ma[],
    ssz_t nb,
    tpsa_t *mc[],
    idx_t select[]
);
typedef void (*madng_ctpsa_map_partial_inverse_fn)(
    ssz_t na,
    const ctpsa_t *ma[],
    ssz_t nb,
    ctpsa_t *mc[],
    idx_t select[]
);

/* vec2fld */
typedef void (*madng_tpsa_scalar_to_map_fn)(
    ssz_t na,
    const tpsa_t *a,
    tpsa_t *mc[]
);
typedef void (*madng_ctpsa_scalar_to_map_fn)(
    ssz_t na,
    const ctpsa_t *a,
    ctpsa_t *mc[]
);

/* fld2vec */
typedef void (*madng_tpsa_map_to_scalar_fn)(
    ssz_t na,
    const tpsa_t *ma[],
    tpsa_t *c
);
typedef void (*madng_ctpsa_map_to_scalar_fn)(
    ssz_t na,
    const ctpsa_t *ma[],
    ctpsa_t *c
);

/* translate */
typedef void (*madng_tpsa_map_translate_fn)(
    ssz_t na,
    const tpsa_t *ma[],
    ssz_t nb,
    const num_t tb[],
    tpsa_t *mc[]
);
typedef void (*madng_ctpsa_map_translate_fn)(
    ssz_t na,
    const ctpsa_t *ma[],
    ssz_t nb,
    const cpx_t tb[],
    ctpsa_t *mc[]
);

/* eval */
typedef void (*madng_tpsa_map_eval_fn)(
    ssz_t na,
    const tpsa_t *ma[],
    ssz_t nb,
    const num_t tb[],
    num_t tc[]
);
typedef void (*madng_ctpsa_map_eval_fn)(
    ssz_t na,
    const ctpsa_t *ma[],
    ssz_t nb,
    const cpx_t tb[],
    cpx_t tc[]
);


/* ------------------------------------------------------------------------- */
/* Protected calls                                                           */
/* ------------------------------------------------------------------------- */

int madng_tpsa_protected_unary_call(
    madng_tpsa_unary_fn function,
    const tpsa_t *input,
    tpsa_t *output
);
int madng_tpsa_protected_binary_call(
    madng_tpsa_binary_fn function,
    const tpsa_t *left,
    const tpsa_t *right,
    tpsa_t *output
);
int madng_tpsa_protected_two_output_call(
    madng_tpsa_two_output_fn function,
    const tpsa_t *input,
    tpsa_t *first_output,
    tpsa_t *second_output
);
int madng_tpsa_protected_map_binary_call(
    madng_tpsa_map_binary_fn function,
    ssz_t na,
    const tpsa_t *ma[],
    ssz_t nb,
    const tpsa_t *mb[],
    tpsa_t *mc[]
);
int madng_tpsa_protected_map_pair_call(
    madng_tpsa_map_pair_fn function,
    ssz_t na,
    const tpsa_t *ma[],
    const tpsa_t *mb[],
    tpsa_t *mc[]
);
int madng_tpsa_protected_map_inverse_call(
    madng_tpsa_map_inverse_fn function,
    ssz_t na,
    const tpsa_t *ma[],
    ssz_t nb,
    tpsa_t *mc[]
);
int madng_tpsa_protected_map_partial_inverse_call(
    madng_tpsa_map_partial_inverse_fn function,
    ssz_t na,
    const tpsa_t *ma[],
    ssz_t nb,
    tpsa_t *mc[],
    idx_t select[]
);
int madng_tpsa_protected_scalar_to_map_call(
    madng_tpsa_scalar_to_map_fn function,
    ssz_t na,
    const tpsa_t *a,
    tpsa_t *mc[]
);
int madng_tpsa_protected_map_to_scalar_call(
    madng_tpsa_map_to_scalar_fn function,
    ssz_t na,
    const tpsa_t *ma[],
    tpsa_t *c
);
int madng_tpsa_protected_map_translate_call(
    madng_tpsa_map_translate_fn function,
    ssz_t na,
    const tpsa_t *ma[],
    ssz_t nb,
    const num_t tb[],
    tpsa_t *mc[]
);
int madng_tpsa_protected_map_eval_call(
    madng_tpsa_map_eval_fn function,
    ssz_t na,
    const tpsa_t *ma[],
    ssz_t nb,
    const num_t tb[],
    num_t tc[]
);
int madng_ctpsa_protected_map_binary_call(
    madng_ctpsa_map_binary_fn function,
    ssz_t na,
    const ctpsa_t *ma[],
    ssz_t nb,
    const ctpsa_t *mb[],
    ctpsa_t *mc[]
);
int madng_ctpsa_protected_map_pair_call(
    madng_ctpsa_map_pair_fn function,
    ssz_t na,
    const ctpsa_t *ma[],
    const ctpsa_t *mb[],
    ctpsa_t *mc[]
);
int madng_ctpsa_protected_map_inverse_call(
    madng_ctpsa_map_inverse_fn function,
    ssz_t na,
    const ctpsa_t *ma[],
    ssz_t nb,
    ctpsa_t *mc[]
);
int madng_ctpsa_protected_map_partial_inverse_call(
    madng_ctpsa_map_partial_inverse_fn function,
    ssz_t na,
    const ctpsa_t *ma[],
    ssz_t nb,
    ctpsa_t *mc[],
    idx_t select[]
);
int madng_ctpsa_protected_scalar_to_map_call(
    madng_ctpsa_scalar_to_map_fn function,
    ssz_t na,
    const ctpsa_t *a,
    ctpsa_t *mc[]
);
int madng_ctpsa_protected_map_to_scalar_call(
    madng_ctpsa_map_to_scalar_fn function,
    ssz_t na,
    const ctpsa_t *ma[],
    ctpsa_t *c
);
int madng_ctpsa_protected_map_translate_call(
    madng_ctpsa_map_translate_fn function,
    ssz_t na,
    const ctpsa_t *ma[],
    ssz_t nb,
    const cpx_t tb[],
    ctpsa_t *mc[]
);
int madng_ctpsa_protected_map_eval_call(
    madng_ctpsa_map_eval_fn function,
    ssz_t na,
    const ctpsa_t *ma[],
    ssz_t nb,
    const cpx_t tb[],
    cpx_t tc[]
);


/* ------------------------------------------------------------------------- */
/* Captured MAD-NG diagnostics                                               */
/* ------------------------------------------------------------------------- */

const char *madng_tpsa_last_error_location(void);
const char *madng_tpsa_last_error_message(void);

#endif // _MADNG_PANIC_H
