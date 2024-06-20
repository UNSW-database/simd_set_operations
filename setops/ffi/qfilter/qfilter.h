#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// QFilter:
int intersect_qfilter_uint_b4(
            const int *set_a, int size_a,
            const int *set_b, int size_b,
            int *set_c);

int intersect_qfilter_uint_b4_v2(
            const int *set_a, int size_a,
            const int *set_b, int size_b,
            int *set_c);

// QFilter+BSR:
int intersect_qfilter_bsr_b4(
            const int* bases_a, const int* states_a, int size_a,
            const int* bases_b, const int* states_b, int size_b,
            int* bases_c, int* states_c);

int intersect_qfilter_bsr_b4_v2(
            const int* bases_a, const int* states_a, int size_a,
            const int* bases_b, const int* states_b, int size_b,
            int* bases_c, int* states_c);

// BMiss:
int intersect_bmiss_uint_b4(
            const int *set_a, int size_a,
            const int *set_b, int size_b,
            int *set_c);

// BMiss+STTNI (block size = 8):
int intersect_bmiss_uint_sttni_b8(
            const int *set_a, int size_a,
            const int *set_b, int size_b,
            int *set_c);
#ifdef __cplusplus
}
#endif
