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

/* ------------------------------------------------------------------------- */
/* Captured MAD-NG diagnostics                                               */
/* ------------------------------------------------------------------------- */

const char *madng_tpsa_last_error_location(void);
const char *madng_tpsa_last_error_message(void);

#include <madng_panic_generated.h> // generated in the build directory

#endif // _MADNG_PANIC_H
