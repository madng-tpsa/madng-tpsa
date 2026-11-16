"""C function definitions for the MADNG TPSA library."""

DEFINITIONS = {
    'minv': (
        """
        void mad_tpsa_minv(
            ssz_t na,
            const tpsa_t *ma[],
            ssz_t nb,
            tpsa_t *mc[]
        );
        """,
        {},
    ),
    'compose': (
        """
        void mad_tpsa_compose(
            ssz_t na,
            const tpsa_t *ma[],
            ssz_t nb,
            const tpsa_t *mb[],
            tpsa_t *mc[]
        );
        """,
        {'array_sizes': {'mc': 'na'}},
    ),
    'liebra': (
        """
        void mad_tpsa_liebra(
            ssz_t na,
            const tpsa_t *ma[],
            const tpsa_t *mb[],
            tpsa_t *mc[]
        );
        """,
        {},
    ),
    'vec2fld': (
        """
        void mad_tpsa_vec2fld(
            ssz_t na,
            const tpsa_t *a,
            tpsa_t *mc[]
        );
        """,
        {},
    ),
}
