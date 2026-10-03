# ruff: noqa: E501

"""CFFI ABI bindings for the packaged ``libmadng_tpsa`` shared library.

The declarations below are the small MAD-NG GTPSA subset used by the Python
wrapper. TPSA and descriptor objects stay opaque at this layer. MAD-NG uses
1-based variable and parameter indices.
"""

from __future__ import annotations

from typing import Any

import cffi

from .paths import core_library

CDEF = """
    extern unsigned char mad_tpsa_dflt;
    extern unsigned char mad_tpsa_same;

    void* mad_desc_newv(int nv, unsigned char mo);
    void* mad_desc_newvp(int nv, unsigned char mo, int np, unsigned char po);
    void* mad_desc_newvpo(int nv, unsigned char mo, int np, unsigned char po, const unsigned char* no);
    void mad_desc_del(const void* d);
    int mad_desc_getnv(const void* d, unsigned char* mo_, int* np_, unsigned char* po_);
    unsigned char mad_desc_maxord(const void* d, int n, unsigned char* no);
    _Bool mad_desc_isvalidm(const void* d, int n, const unsigned char* m);
    int mad_desc_idxm(const void* d, int n, const unsigned char* m);

    void* mad_tpsa_newd(void* desc, unsigned char mo);
    const void* mad_tpsa_desc(const void* t);
    unsigned char mad_tpsa_ord(const void* t, _Bool hi);
    void mad_tpsa_del(void* t);
    void mad_tpsa_setvar(void* t, double v, int iv, double scl);
    void mad_tpsa_setprm(void* t, double v, int ip);
    void mad_tpsa_setval(void* t, double v);
    void mad_tpsa_clear(void* t);
    int mad_tpsa_isnul(const void* t);
    int mad_tpsa_isval(const void* t);
    double mad_tpsa_geti(void* t, int i);
    double mad_tpsa_getm(void* t, int n, const unsigned char* m);
    void mad_tpsa_seti(void* t, int i, double a, double b);
    void mad_tpsa_setm(void* t, int n, const unsigned char* m, double a, double b);
    void mad_tpsa_copy(const void* t, void* r);
    int mad_tpsa_cycle(void* t, int i, int n, unsigned char* m, double* v);
    void mad_tpsa_getord(const void* t, void* r, unsigned char ord);
    void mad_tpsa_cutord(const void* t, void* r, int ord);
    void mad_tpsa_clrord(void* t, unsigned char ord);

    void mad_tpsa_add(const void* a, const void* b, void* c);
    void mad_tpsa_sub(const void* a, const void* b, void* c);
    void mad_tpsa_mul(const void* a, const void* b, void* c);
    void mad_tpsa_div(const void* a, const void* b, void* c);
    int mad_tpsa_equ(const void* a, const void* b, double tol);
    void mad_tpsa_pow(const void* a, const void* b, void* c);
    void mad_tpsa_powi(const void* a, int n, void* c);
    void mad_tpsa_pown(const void* a, double v, void* c);

    void* mad_ctpsa_newd(const void* desc, unsigned char mo);
    void mad_ctpsa_del(const void* t);
    const void* mad_ctpsa_desc(const void* t);
    unsigned char mad_ctpsa_ord(const void* t, _Bool hi);
    void mad_ctpsa_setvar_r(
        void* t, double v_re, double v_im, int iv, double scl_re, double scl_im
    );
    void mad_ctpsa_setprm_r(void* t, double v_re, double v_im, int ip);
    void mad_ctpsa_setval_r(void* t, double v_re, double v_im);
    void mad_ctpsa_clear(void* t);
    int mad_ctpsa_isnul(const void* t);
    int mad_ctpsa_isval(const void* t);
    void mad_ctpsa_copy(const void* t, void* r);
    void mad_ctpsa_cplx(const void* re, const void* im, void* r);
    void mad_ctpsa_real(const void* t, void* r);
    void mad_ctpsa_imag(const void* t, void* r);
    int mad_ctpsa_equ(const void* a, const void* b, double tol);
    void mad_ctpsa_geti_r(const void* t, int i, double _Complex* r);
    void mad_ctpsa_getm_r(const void* t, int n, const unsigned char* m, double _Complex* r);
    void mad_ctpsa_seti_r(void* t, int i, double a_re, double a_im, double b_re, double b_im);
    void mad_ctpsa_setm_r(void* t, int n, const unsigned char* m, double a_re, double a_im, double b_re, double b_im);
    int mad_ctpsa_cycle(const void* t, int i, int n, unsigned char* m, double _Complex* v);
    void mad_ctpsa_getord(const void* t, void* r, unsigned char ord);
    void mad_ctpsa_cutord(const void* t, void* r, int ord);
    void mad_ctpsa_clrord(void* t, unsigned char ord);

    void mad_ctpsa_add(const void* a, const void* b, void* c);
    void mad_ctpsa_sub(const void* a, const void* b, void* c);
    void mad_ctpsa_mul(const void* a, const void* b, void* c);
    void mad_ctpsa_div(const void* a, const void* b, void* c);
    void mad_ctpsa_pow(const void* a, const void* b, void* c);
    void mad_ctpsa_powi(const void* a, int n, void* c);
    void mad_ctpsa_pown_r(const void* a, double v_re, double v_im, void* c);
    void mad_ctpsa_addt(const void* a, const void* b, void* c);
    void mad_ctpsa_subt(const void* a, const void* b, void* c);
    void mad_ctpsa_tsub(const void* a, const void* b, void* c);
    void mad_ctpsa_mult(const void* a, const void* b, void* c);
    void mad_ctpsa_divt(const void* a, const void* b, void* c);
    void mad_ctpsa_tdiv(const void* a, const void* b, void* c);
    void mad_ctpsa_powt(const void* a, const void* b, void* c);
    void mad_ctpsa_tpow(const void* a, const void* b, void* c);
    void mad_ctpsa_axpb_r(double a_re, double a_im, const void* x, double b_re, double b_im, void* r);
    void mad_ctpsa_scl_r(const void* a, double v_re, double v_im, void* c);
    void mad_ctpsa_divn_r(const void* a, double v_re, double v_im, void* c);
    void mad_ctpsa_inv_r(const void* a, double v_re, double v_im, void* c);

    double mad_ctpsa_nrm(const void* a);
    void mad_ctpsa_unit(const void* a, void* c);
    void mad_ctpsa_conj(const void* a, void* c);
    void mad_ctpsa_sqrt(const void* a, void* c);
    void mad_ctpsa_exp(const void* a, void* c);
    void mad_ctpsa_log(const void* a, void* c);
    void mad_ctpsa_sin(const void* a, void* c);
    void mad_ctpsa_cos(const void* a, void* c);
    void mad_ctpsa_tan(const void* a, void* c);
    void mad_ctpsa_sinh(const void* a, void* c);
    void mad_ctpsa_cosh(const void* a, void* c);
    void mad_ctpsa_tanh(const void* a, void* c);
    void mad_ctpsa_asin(const void* a, void* c);
    void mad_ctpsa_acos(const void* a, void* c);
    void mad_ctpsa_atan(const void* a, void* c);
    void mad_ctpsa_asinh(const void* a, void* c);
    void mad_ctpsa_acosh(const void* a, void* c);
    void mad_ctpsa_atanh(const void* a, void* c);
    void mad_ctpsa_erf(const void* a, void* c);
    void mad_ctpsa_erfc(const void* a, void* c);
    void mad_ctpsa_erfcx(const void* a, void* c);
    void mad_ctpsa_erfi(const void* a, void* c);
    void mad_ctpsa_wf(const void* a, void* c);
    void mad_ctpsa_integ(const void* a, void* c, int iv);
    void mad_ctpsa_deriv(const void* a, void* c, int iv);
    void mad_ctpsa_derivm(const void* a, void* c, int n, const unsigned char* m);
    void mad_ctpsa_poisbra(const void* a, const void* b, void* c, int nv);

    void mad_tpsa_scl(const void* a, double v, void* c);
    void mad_tpsa_divn(const void* a, double v, void* c);
    void mad_tpsa_inv(const void* a, double v, void* c);

    double mad_tpsa_nrm(const void* a);
    void mad_tpsa_unit(const void* a, void* c);
    void mad_tpsa_abs(const void* a, void* c);
    void mad_tpsa_sqrt(const void* a, void* c);
    void mad_tpsa_exp(const void* a, void* c);
    void mad_tpsa_log(const void* a, void* c);

    void mad_tpsa_sin(const void* a, void* c);
    void mad_tpsa_cos(const void* a, void* c);
    void mad_tpsa_tan(const void* a, void* c);
    void mad_tpsa_sinc(const void* a, void* c);
    void mad_tpsa_sincos(const void* a, void* s, void* c);
    void mad_tpsa_sincosq(const void* a, void* s, void* c);
    void mad_tpsa_sincosmq(const void* a, void* s, void* c);

    void mad_tpsa_sinh(const void* a, void* c);
    void mad_tpsa_cosh(const void* a, void* c);
    void mad_tpsa_tanh(const void* a, void* c);
    void mad_tpsa_sinhc(const void* a, void* c);
    void mad_tpsa_sincosh(const void* a, void* s, void* c);
    void mad_tpsa_sincoshq(const void* a, void* s, void* c);
    void mad_tpsa_sincoshmq(const void* a, void* s, void* c);

    void mad_tpsa_asin(const void* a, void* c);
    void mad_tpsa_acos(const void* a, void* c);
    void mad_tpsa_atan(const void* a, void* c);
    void mad_tpsa_asinh(const void* a, void* c);
    void mad_tpsa_acosh(const void* a, void* c);
    void mad_tpsa_atanh(const void* a, void* c);

    void mad_tpsa_erf(const void* a, void* c);
    void mad_tpsa_erfc(const void* a, void* c);
    void mad_tpsa_erfcx(const void* a, void* c);
    void mad_tpsa_erfi(const void* a, void* c);
    void mad_tpsa_wf(const void* a, void* c);

    void mad_tpsa_atan2(const void* y, const void* x, void* r);
    void mad_tpsa_hypot(const void* x, const void* y, void* r);
    void mad_tpsa_hypot3(const void* x, const void* y, const void* z, void* r);
    void mad_tpsa_axpb(double a, const void* x, double b, void* r);
    void mad_tpsa_integ(const void* a, void* c, int iv);
    void mad_tpsa_deriv(const void* a, void* c, int iv);
    void mad_tpsa_derivm(const void* a, void* c, int n, const unsigned char* m);
    void mad_tpsa_poisbra(const void* a, const void* b, void* c, int nv);

    /*
     * Native real map operations.
     *
     * MAD-NG's ssz_t and idx_t are both int32_t. The bindings already use
     * plain int for these ABI arguments.
     */
    void mad_tpsa_vec2fld(int na, const void* a, void* mc[]);
    void mad_tpsa_fld2vec(int na, const void* ma[], void* c);
    void mad_tpsa_liebra(int na, const void* ma[], const void* mb[], void* mc[]);
    void mad_tpsa_exppb(int na, const void* ma[], int nb, const void* mb[], void* mc[]);
    void mad_tpsa_logpb(int na, const void* ma[], const void* mb[], void* mc[]);
    unsigned char mad_tpsa_mord(int na, const void* ma[], _Bool hi);
    double mad_tpsa_mnrm(int na, const void* ma[]);
    void mad_tpsa_minv(int na, const void* ma[], int nb, void* mc[]);
    void mad_tpsa_pminv(int na, const void* ma[], int nb, void* mc[], int select[]);
    void mad_tpsa_compose(int na, const void* ma[], int nb, const void* mb[], void* mc[]);
    void mad_tpsa_translate(int na, const void* ma[], int nb, const double tb[], void* mc[]);
    void mad_tpsa_eval(int na, const void* ma[], int nb, const double tb[], double tc[]);

    /* Native complex map operations. */
    void mad_ctpsa_vec2fld(int na, const void* a, void* mc[]);
    void mad_ctpsa_fld2vec(int na, const void* ma[], void* c);
    void mad_ctpsa_liebra(int na, const void* ma[], const void* mb[], void* mc[]);
    void mad_ctpsa_exppb(int na, const void* ma[], int nb, const void* mb[], void* mc[]);
    void mad_ctpsa_logpb(int na, const void* ma[], const void* mb[], void* mc[]);
    unsigned char mad_ctpsa_mord(int na, const void* ma[], _Bool hi);
    double mad_ctpsa_mnrm(int na, const void* ma[]);
    void mad_ctpsa_minv(int na, const void* ma[], int nb, void* mc[]);
    void mad_ctpsa_pminv(int na, const void* ma[], int nb, void* mc[], int select[]);
    void mad_ctpsa_compose(int na, const void* ma[], int nb, const void* mb[], void* mc[]);
    void mad_ctpsa_translate(int na, const void* ma[], int nb, const double _Complex tb[], void* mc[]);
    void mad_ctpsa_eval(int na, const void* ma[], int nb, const double _Complex tb[], double _Complex tc[]);

    /* protected calls for unary, binary, and two-output functions */
    typedef void (*madng_tpsa_unary_fn)(const void* input, void* output);
    typedef void (*madng_tpsa_binary_fn)(const void* left, const void* right, void* output);
    typedef void (*madng_tpsa_two_output_fn)(const void* input, void* first_output, void* second_output);
    int madng_tpsa_protected_unary_call(madng_tpsa_unary_fn function, const void* input, void* output);
    int madng_tpsa_protected_binary_call(madng_tpsa_binary_fn function, const void* left, const void* right, void* output);
    int madng_tpsa_protected_two_output_call(madng_tpsa_two_output_fn function, const void* input, void* first_output, void* second_output);

    /* compose, exppb */
    typedef void (*madng_tpsa_map_binary_fn)(int na, const void* ma[], int nb, const void* mb[], void* mc[]);
    typedef void (*madng_ctpsa_map_binary_fn)(int na, const void* ma[], int nb, const void* mb[], void* mc[]);
    int madng_tpsa_protected_map_binary_call(madng_tpsa_map_binary_fn function, int na, const void* ma[], int nb, const void* mb[], void* mc[]);
    int madng_ctpsa_protected_map_binary_call(madng_ctpsa_map_binary_fn function, int na, const void* ma[], int nb, const void* mb[], void* mc[]);

    /* liebra, logpb */
    typedef void (*madng_tpsa_map_pair_fn)(int na, const void* ma[], const void* mb[], void* mc[]);
    typedef void (*madng_ctpsa_map_pair_fn)(int na, const void* ma[], const void* mb[], void* mc[]);
    int madng_tpsa_protected_map_pair_call(madng_tpsa_map_pair_fn function, int na, const void* ma[], const void* mb[], void* mc[]);
    int madng_ctpsa_protected_map_pair_call(madng_ctpsa_map_pair_fn function, int na, const void* ma[], const void* mb[], void* mc[]);

    /* minv */
    typedef void (*madng_tpsa_map_inverse_fn)(int na, const void* ma[], int nb, void* mc[]);
    typedef void (*madng_ctpsa_map_inverse_fn)(int na, const void* ma[], int nb, void* mc[]);
    int madng_tpsa_protected_map_inverse_call(madng_tpsa_map_inverse_fn function, int na, const void* ma[], int nb, void* mc[]);
    int madng_ctpsa_protected_map_inverse_call(madng_ctpsa_map_inverse_fn function, int na, const void* ma[], int nb, void* mc[]);

    /* pminv */
    typedef void (*madng_tpsa_map_partial_inverse_fn)(int na, const void* ma[], int nb, void* mc[], int select[]);
    typedef void (*madng_ctpsa_map_partial_inverse_fn)(int na, const void* ma[], int nb, void* mc[], int select[]);
    int madng_tpsa_protected_map_partial_inverse_call(madng_tpsa_map_partial_inverse_fn function, int na, const void* ma[], int nb, void* mc[], int select[]);
    int madng_ctpsa_protected_map_partial_inverse_call(madng_ctpsa_map_partial_inverse_fn function, int na, const void* ma[], int nb, void* mc[], int select[]);

    /* vec2fld */
    typedef void (*madng_tpsa_scalar_to_map_fn)(int na, const void* a, void* mc[]);
    typedef void (*madng_ctpsa_scalar_to_map_fn)(int na, const void* a, void* mc[]);
    int madng_tpsa_protected_scalar_to_map_call(madng_tpsa_scalar_to_map_fn function, int na, const void* a, void* mc[]);
    int madng_ctpsa_protected_scalar_to_map_call(madng_ctpsa_scalar_to_map_fn function, int na, const void* a, void* mc[]);

    /* fld2vec */
    typedef void (*madng_tpsa_map_to_scalar_fn)(int na, const void* ma[], void* c);
    typedef void (*madng_ctpsa_map_to_scalar_fn)(int na, const void* ma[], void* c);
    int madng_tpsa_protected_map_to_scalar_call(madng_tpsa_map_to_scalar_fn function, int na, const void* ma[], void* c);
    int madng_ctpsa_protected_map_to_scalar_call(madng_ctpsa_map_to_scalar_fn function, int na, const void* ma[], void* c);

    /* translate */
    typedef void (*madng_tpsa_map_translate_fn)(int na, const void* ma[], int nb, const double tb[], void* mc[]);
    typedef void (*madng_ctpsa_map_translate_fn)(int na, const void* ma[], int nb, const double _Complex tb[], void* mc[]);
    int madng_tpsa_protected_map_translate_call(madng_tpsa_map_translate_fn function, int na, const void* ma[], int nb, const double tb[], void* mc[]);
    int madng_ctpsa_protected_map_translate_call(madng_ctpsa_map_translate_fn function, int na, const void* ma[], int nb, const double _Complex tb[], void* mc[]);

    /* eval */
    typedef void (*madng_tpsa_map_eval_fn)(int na, const void* ma[], int nb, const double tb[], double tc[]);
    typedef void (*madng_ctpsa_map_eval_fn)(int na, const void* ma[], int nb, const double _Complex tb[], double _Complex tc[]);
    int madng_tpsa_protected_map_eval_call(madng_tpsa_map_eval_fn function, int na, const void* ma[], int nb, const double tb[], double tc[]);
    int madng_ctpsa_protected_map_eval_call(madng_ctpsa_map_eval_fn function, int na, const void* ma[], int nb, const double _Complex tb[], double _Complex tc[]);

    const char* madng_tpsa_last_error_location(void);
    const char* madng_tpsa_last_error_message(void);

    int madng_tpsa_check_tpsa_compatibility(const void* left, const void* right);
    int madng_tpsa_tpsa_single_monomial(const void* series, int monomial_length, unsigned char* monomial_orders);
"""

ffi = cffi.FFI()
ffi.cdef(CDEF)
lib: Any = ffi.dlopen(core_library())
