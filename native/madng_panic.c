/*
    Convert libgtpsa's non-local fatal errors into status codes.

    libgtpsa reports an invalid TPSA operation through mad_error(), which is
    noreturn: it normally terminates the process. Python bindings instead need
    to return to Python and raise an exception. setjmp() records a recovery
    point before calling libgtpsa. Our mad_error handler copies the diagnostic,
    then longjmp() resumes at that recovery point, where the protected call
    returns a non-zero status.
*/

#include <setjmp.h>
#include <stdio.h>

#include "madng_panic.h"

enum { MADNG_TPSA_ERROR_TEXT_SIZE = 1024 };

/*
    Each thread has one active protected call. libgtpsa invokes the installed
    handler synchronously, so the handler can copy its error and longjmp back
    to the setjmp below. Protected calls must not nest on a thread.
*/
static thread_local jmp_buf madng_panic_environment;
static thread_local madng_tpsa_error_handler madng_panic_previous_error_handler;
static thread_local char madng_panic_last_location[MADNG_TPSA_ERROR_TEXT_SIZE];
static thread_local char madng_panic_last_message[MADNG_TPSA_ERROR_TEXT_SIZE];

static void madng_panic_copy_error_text(
    char destination[static MADNG_TPSA_ERROR_TEXT_SIZE],
    const char *source
) {
    snprintf(destination, MADNG_TPSA_ERROR_TEXT_SIZE, "%s", source ? source : "");
}

static void madng_panic_longjmp(const char *location, const char *message) {
    /* The message passed by mad_error is stack-owned, so retain it before jumping. */
    madng_panic_copy_error_text(madng_panic_last_location, location);
    madng_panic_copy_error_text(madng_panic_last_message, message);
    longjmp(madng_panic_environment, 1);
}

static void madng_panic_restore_error_handler(void) {
    madng_tpsa_set_error_handler(madng_panic_previous_error_handler);
}


/*
    Every protected wrapper has the same control flow. Only the actual MAD-NG
    call differs.

    The macro deliberately returns from the enclosing wrapper.
*/
#define MADNG_PROTECTED_CALL(call)                                                           \
do {                                                                                         \
    /* Always restore the caller's handler, whether MAD-NG returns or panics. */             \
    madng_panic_previous_error_handler = madng_tpsa_set_error_handler(madng_panic_longjmp);  \
    if (setjmp(madng_panic_environment) == 0) {                                              \
        call;                                                                                \
        madng_panic_restore_error_handler();                                                 \
        return 0;                                                                            \
    }                                                                                        \
    madng_panic_restore_error_handler();                                                     \
    return 1;                                                                                \
} while (0)


int madng_tpsa_protected_unary_call(
    madng_tpsa_unary_fn function,
    const tpsa_t *input,
    tpsa_t *output
) {MADNG_PROTECTED_CALL(function(input, output));}

int madng_tpsa_protected_binary_call(
    madng_tpsa_binary_fn function,
    const tpsa_t *left,
    const tpsa_t *right,
    tpsa_t *output
) {MADNG_PROTECTED_CALL(function(left, right, output));}

int madng_tpsa_protected_two_output_call(
    madng_tpsa_two_output_fn function,
    const tpsa_t *input,
    tpsa_t *first_output,
    tpsa_t *second_output
) {MADNG_PROTECTED_CALL(function(input, first_output, second_output));}

int madng_tpsa_protected_map_binary_call(
    madng_tpsa_map_binary_fn function,
    ssz_t na,
    const tpsa_t *ma[],
    ssz_t nb,
    const tpsa_t *mb[],
    tpsa_t *mc[]
) {MADNG_PROTECTED_CALL(function(na, ma, nb, mb, mc));}

int madng_tpsa_protected_map_pair_call(
    madng_tpsa_map_pair_fn function,
    ssz_t na,
    const tpsa_t *ma[],
    const tpsa_t *mb[],
    tpsa_t *mc[]
) {MADNG_PROTECTED_CALL(function(na, ma, mb, mc));}

int madng_tpsa_protected_map_inverse_call(
    madng_tpsa_map_inverse_fn function,
    ssz_t na,
    const tpsa_t *ma[],
    ssz_t nb,
    tpsa_t *mc[]
) {MADNG_PROTECTED_CALL(function(na, ma, nb, mc));}

int madng_tpsa_protected_map_partial_inverse_call(
    madng_tpsa_map_partial_inverse_fn function,
    ssz_t na,
    const tpsa_t *ma[],
    ssz_t nb,
    tpsa_t *mc[],
    idx_t select[]
) {MADNG_PROTECTED_CALL(function(na, ma, nb, mc, select));}

int madng_tpsa_protected_scalar_to_map_call(
    madng_tpsa_scalar_to_map_fn function,
    ssz_t na,
    const tpsa_t *a,
    tpsa_t *mc[]
) {MADNG_PROTECTED_CALL(function(na, a, mc));}

int madng_tpsa_protected_map_to_scalar_call(
    madng_tpsa_map_to_scalar_fn function,
    ssz_t na,
    const tpsa_t *ma[],
    tpsa_t *c
) {MADNG_PROTECTED_CALL(function(na, ma, c));}

int madng_tpsa_protected_map_translate_call(
    madng_tpsa_map_translate_fn function,
    ssz_t na,
    const tpsa_t *ma[],
    ssz_t nb,
    const num_t tb[],
    tpsa_t *mc[]
) {MADNG_PROTECTED_CALL(function(na, ma, nb, tb, mc));}

int madng_tpsa_protected_map_eval_call(
    madng_tpsa_map_eval_fn function,
    ssz_t na,
    const tpsa_t *ma[],
    ssz_t nb,
    const num_t tb[],
    num_t tc[]
) {MADNG_PROTECTED_CALL(function(na, ma, nb, tb, tc));}

int madng_ctpsa_protected_map_binary_call(
    madng_ctpsa_map_binary_fn function,
    ssz_t na,
    const ctpsa_t *ma[],
    ssz_t nb,
    const ctpsa_t *mb[],
    ctpsa_t *mc[]
) {MADNG_PROTECTED_CALL(function(na, ma, nb, mb, mc));}

int madng_ctpsa_protected_map_pair_call(
    madng_ctpsa_map_pair_fn function,
    ssz_t na,
    const ctpsa_t *ma[],
    const ctpsa_t *mb[],
    ctpsa_t *mc[]
) {MADNG_PROTECTED_CALL(function(na, ma, mb, mc));}

int madng_ctpsa_protected_map_inverse_call(
    madng_ctpsa_map_inverse_fn function,
    ssz_t na,
    const ctpsa_t *ma[],
    ssz_t nb,
    ctpsa_t *mc[]
) {MADNG_PROTECTED_CALL(function(na, ma, nb, mc));}

int madng_ctpsa_protected_map_partial_inverse_call(
    madng_ctpsa_map_partial_inverse_fn function,
    ssz_t na,
    const ctpsa_t *ma[],
    ssz_t nb,
    ctpsa_t *mc[],
    idx_t select[]
) {MADNG_PROTECTED_CALL(function(na, ma, nb, mc, select));}

int madng_ctpsa_protected_scalar_to_map_call(
    madng_ctpsa_scalar_to_map_fn function,
    ssz_t na,
    const ctpsa_t *a,
    ctpsa_t *mc[]
) {MADNG_PROTECTED_CALL(function(na, a, mc));}

int madng_ctpsa_protected_map_to_scalar_call(
    madng_ctpsa_map_to_scalar_fn function,
    ssz_t na,
    const ctpsa_t *ma[],
    ctpsa_t *c
) {MADNG_PROTECTED_CALL(function(na, ma, c));}

int madng_ctpsa_protected_map_translate_call(
    madng_ctpsa_map_translate_fn function,
    ssz_t na,
    const ctpsa_t *ma[],
    ssz_t nb,
    const cpx_t tb[],
    ctpsa_t *mc[]
) {MADNG_PROTECTED_CALL(function(na, ma, nb, tb, mc));}

int madng_ctpsa_protected_map_eval_call(
    madng_ctpsa_map_eval_fn function,
    ssz_t na,
    const ctpsa_t *ma[],
    ssz_t nb,
    const cpx_t tb[],
    cpx_t tc[]
) {MADNG_PROTECTED_CALL(function(na, ma, nb, tb, tc));}


const char *madng_tpsa_last_error_location(void) { return madng_panic_last_location; }
const char *madng_tpsa_last_error_message(void) { return madng_panic_last_message; }
