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
static _Thread_local jmp_buf madng_panic_environment;
static _Thread_local madng_tpsa_error_handler madng_panic_previous_error_handler;
static _Thread_local char madng_panic_last_location[MADNG_TPSA_ERROR_TEXT_SIZE];
static _Thread_local char madng_panic_last_message[MADNG_TPSA_ERROR_TEXT_SIZE];

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

int madng_tpsa_protected_unary_call(
    madng_tpsa_unary_fn function,
    const tpsa_t *input,
    tpsa_t *output
) {
    /* Always restore the caller's handler, whether MAD-NG returns or panics. */
    madng_panic_previous_error_handler = madng_tpsa_set_error_handler(madng_panic_longjmp);

    if (setjmp(madng_panic_environment) == 0) {
        function(input, output);
        madng_panic_restore_error_handler();
        return 0;
    }

    madng_panic_restore_error_handler();
    return 1;
}

int madng_tpsa_protected_binary_call(
    madng_tpsa_binary_fn function,
    const tpsa_t *left,
    const tpsa_t *right,
    tpsa_t *output
) {
    madng_panic_previous_error_handler = madng_tpsa_set_error_handler(madng_panic_longjmp);

    if (setjmp(madng_panic_environment) == 0) {
        function(left, right, output);
        madng_panic_restore_error_handler();
        return 0;
    }

    madng_panic_restore_error_handler();
    return 1;
}

int madng_tpsa_protected_two_output_call(
    madng_tpsa_two_output_fn function,
    const tpsa_t *input,
    tpsa_t *first_output,
    tpsa_t *second_output
) {
    madng_panic_previous_error_handler = madng_tpsa_set_error_handler(madng_panic_longjmp);

    if (setjmp(madng_panic_environment) == 0) {
        function(input, first_output, second_output);
        madng_panic_restore_error_handler();
        return 0;
    }

    madng_panic_restore_error_handler();
    return 1;
}

const char *madng_tpsa_last_error_location(void) { return madng_panic_last_location; }

const char *madng_tpsa_last_error_message(void) { return madng_panic_last_message; }
