/*
    Standalone MAD-NG logging implementation.

    This replaces MAD-NG's mad_log.c so fatal MAD-NG errors can be intercepted.
    Without an installed handler original behaviour is preserved.
*/

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "mad_log.h"
#include "madng_log.h"

int mad_warn_count = 0;
int mad_trace_level = 0;
int mad_trace_location = 0;

static _Thread_local madng_tpsa_error_handler madng_tpsa_current_error_handler = NULL;

madng_tpsa_error_handler madng_tpsa_set_error_handler(madng_tpsa_error_handler handler) {
    madng_tpsa_error_handler previous_handler = madng_tpsa_current_error_handler;
    madng_tpsa_current_error_handler = handler;
    return previous_handler;
}

/* MAD-NG's standalone mad_error, modified to call an installed error handler. */
void(mad_error)(str_t fn, str_t fmt, ...) {
    va_list arguments;
    va_start(arguments, fmt);

    if (madng_tpsa_current_error_handler) {
        char message[1024];
        vsnprintf(message, sizeof(message), fmt, arguments);
        va_end(arguments);
        madng_tpsa_current_error_handler(fn, message);
        exit(EXIT_FAILURE); /* A handler must not return. */
    }

    fflush(stdout);
    fprintf(stderr, fn ? "error: %s: " : "error: ", fn);
    vfprintf(stderr, fmt, arguments);
    va_end(arguments);
    fputc('\n', stderr);
    exit(EXIT_FAILURE);
}

/* MAD-NG's standalone mad_warn, unchanged but definition required due to swapping out mad_log.c */
void(mad_warn)(str_t fn, str_t fmt, ...) {
    ++mad_warn_count;
    va_list arguments;
    va_start(arguments, fmt);
    fflush(stdout);
    fprintf(stderr, fn ? "warning: %s: " : "warning: ", fn);
    vfprintf(stderr, fmt, arguments);
    va_end(arguments);
    fputc('\n', stderr);
}

/* MAD-NG's standalone mad_trace, unchanged but definition required due to swapping out mad_log.c */
void(mad_trace)(int level, str_t fn, str_t fmt, ...) {
    if (mad_trace_level < level)
        return;
    va_list arguments;
    va_start(arguments, fmt);
    fflush(stdout);
    if (fn)
        fprintf(stderr, "%s", fn);
    vfprintf(stderr, fmt, arguments);
    va_end(arguments);
    fputc('\n', stderr);
}
