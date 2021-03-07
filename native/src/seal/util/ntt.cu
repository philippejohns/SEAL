// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/util/ntt.h"
#include "seal/util/uintarith.h"
#include "seal/util/uintarithsmallmod.h"
#include <algorithm>

#include <cstdio>

using namespace std;

//THIS IS A BAD IDEA
std::uint64_t *values_device = nullptr;


namespace seal
{
    namespace util
    {
        NTTTables::NTTTables(int coeff_count_power, const Modulus &modulus, MemoryPoolHandle pool) : pool_(move(pool))
        {
#ifdef SEAL_DEBUG
            if (!pool_)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            initialize(coeff_count_power, modulus);
        }

        void NTTTables::initialize(int coeff_count_power, const Modulus &modulus)
        {
#ifdef SEAL_DEBUG
            if ((coeff_count_power < get_power_of_two(SEAL_POLY_MOD_DEGREE_MIN)) ||
                coeff_count_power > get_power_of_two(SEAL_POLY_MOD_DEGREE_MAX))
            {
                throw invalid_argument("coeff_count_power out of range");
            }
#endif
            coeff_count_power_ = coeff_count_power;
            coeff_count_ = size_t(1) << coeff_count_power_;
            modulus_ = modulus;
            // We defer parameter checking to try_minimal_primitive_root(...)
            if (!try_minimal_primitive_root(2 * coeff_count_, modulus_, root_))
            {
                throw invalid_argument("invalid modulus");
            }
            if (!try_invert_uint_mod(root_, modulus_, inv_root_))
            {
                throw invalid_argument("invalid modulus");
            }

            // Populate tables with powers of root in specific orders.
            root_powers_ = allocate<MultiplyUIntModOperand>(coeff_count_, pool_);
            MultiplyUIntModOperand root;
            root.set(root_, modulus_);
            uint64_t power = root_;
            for (size_t i = 1; i < coeff_count_; i++)
            {
                root_powers_[reverse_bits(i, coeff_count_power_)].set(power, modulus_);
                power = multiply_uint_mod(power, root, modulus_);
            }

            inv_root_powers_ = allocate<MultiplyUIntModOperand>(coeff_count_, pool_);
            root.set(inv_root_, modulus_);
            power = inv_root_;
            for (size_t i = 1; i < coeff_count_; i++)
            {
                inv_root_powers_[reverse_bits(i - 1, coeff_count_power_) + 1].set(power, modulus_);
                power = multiply_uint_mod(power, root, modulus_);
            }

            // Compute n^(-1) modulo q.
            uint64_t degree_uint = static_cast<uint64_t>(coeff_count_);
            if (!try_invert_uint_mod(degree_uint, modulus_, inv_degree_modulo_.operand))
            {
                throw invalid_argument("invalid modulus");
            }
            inv_degree_modulo_.set_quotient(modulus_);

            mod_arith_lazy_ = ModArithLazy(modulus_);
            ntt_handler_ = NTTHandler(mod_arith_lazy_);

            //added to allocate on device
            int log_n = coeff_count_power;
            size_t n = size_t(1) << log_n;

            cudaMalloc(&root_powers_device, n * sizeof(MultiplyUIntModOperand)); //note: this doesnt get destroyed yet...
            cudaMemcpy(root_powers_device, root_powers_.get(), n * sizeof(MultiplyUIntModOperand), cudaMemcpyHostToDevice); 

            //THIS IS AN PART OF THE GLOBALLY BAD IDEA
            if (values_device == nullptr)
                cudaMalloc(&values_device, n * sizeof(std::uint64_t));
        }

        class NTTTablesCreateIter
        {
        public:
            using value_type = NTTTables;
            using pointer = void;
            using reference = value_type;
            using difference_type = ptrdiff_t;

            // LegacyInputIterator allows reference to be equal to value_type so we can construct
            // the return objects on the fly and return by value.
            using iterator_category = input_iterator_tag;

            // Require default constructor
            NTTTablesCreateIter()
            {}

            // Other constructors
            NTTTablesCreateIter(int coeff_count_power, vector<Modulus> modulus, MemoryPoolHandle pool)
                : coeff_count_power_(coeff_count_power), modulus_(modulus), pool_(move(pool))
            {}

            // Require copy and move constructors and assignments
            NTTTablesCreateIter(const NTTTablesCreateIter &copy) = default;

            NTTTablesCreateIter(NTTTablesCreateIter &&source) = default;

            NTTTablesCreateIter &operator=(const NTTTablesCreateIter &assign) = default;

            NTTTablesCreateIter &operator=(NTTTablesCreateIter &&assign) = default;

            // Dereferencing creates NTTTables and returns by value
            inline value_type operator*() const
            {
                return { coeff_count_power_, modulus_[index_], pool_ };
            }

            // Pre-increment
            inline NTTTablesCreateIter &operator++() noexcept
            {
                index_++;
                return *this;
            }

            // Post-increment
            inline NTTTablesCreateIter operator++(int) noexcept
            {
                NTTTablesCreateIter result(*this);
                index_++;
                return result;
            }

            // Must be EqualityComparable
            inline bool operator==(const NTTTablesCreateIter &compare) const noexcept
            {
                return (compare.index_ == index_) && (coeff_count_power_ == compare.coeff_count_power_);
            }

            inline bool operator!=(const NTTTablesCreateIter &compare) const noexcept
            {
                return !operator==(compare);
            }

            // Arrow operator must be defined
            value_type operator->() const
            {
                return **this;
            }

        private:
            size_t index_ = 0;
            int coeff_count_power_ = 0;
            vector<Modulus> modulus_;
            MemoryPoolHandle pool_;
        };

        void CreateNTTTables(
            int coeff_count_power, const vector<Modulus> &modulus, Pointer<NTTTables> &tables, MemoryPoolHandle pool)
        {
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
            if (!modulus.size())
            {
                throw invalid_argument("invalid modulus");
            }
            // coeff_count_power and modulus will be validated by "allocate"

            NTTTablesCreateIter iter(coeff_count_power, modulus, pool);
            tables = allocate(iter, modulus.size(), pool);
        }

        #include <cstdio>
        //shorthand for the Arithmetic template classs to keep the following function header short
        #define ARITH_SH Arithmetic<std::uint64_t, MultiplyUIntModOperand, MultiplyUIntModOperand>

        __global__ void transform_to_rev_kernel(ARITH_SH arithmetic_,
                std::uint64_t * __restrict__ values, const MultiplyUIntModOperand * __restrict__ roots, size_t gap, size_t m)
        {
            // registers to hold temporary values
            MultiplyUIntModOperand r;
            std::uint64_t u;
            std::uint64_t v;
            // pointers for faster indexing
            std::uint64_t *x = nullptr;
            std::uint64_t *y = nullptr;
            std::size_t offset = 0;

            roots += (m-1);

            std::size_t i = threadIdx.x; //m
            std::size_t j = blockIdx.x; //gap

            //for (std::size_t i = 0; i < m; i++)
            //{
                offset = i * (gap << 1);
                r = roots[i+1];
                x = values + offset;
                y = x + gap;
                //for (std::size_t j = 0; j < gap; j++)
                //{
                    u = arithmetic_.guard(x[j]);
                    v = arithmetic_.mul_root(y[j], r);
                    x[j] = arithmetic_.add(u, v);
                    y[j] = arithmetic_.sub(u, v);
                //}
            //}
        }

        __global__ void transform_to_rev_kernel2(const ARITH_SH arithmetic_,
                std::uint64_t * __restrict__ values, const MultiplyUIntModOperand * __restrict__ roots, size_t gap, size_t m)
        {
            // registers to hold temporary values
            MultiplyUIntModOperand r;
            std::uint64_t u;
            std::uint64_t v;
            // pointers for faster indexing
            std::uint64_t *x = nullptr;
            std::uint64_t *y = nullptr;
            std::size_t offset = 0;

            roots += (m-1);

            std::size_t j = threadIdx.x; //gap
            std::size_t i = blockIdx.x; //m

            //for (std::size_t i = 0; i < m; i++)
            //{
                offset = i * (gap << 1);
                r = roots[i+1];
                x = values + offset;
                y = x + gap;
                //for (std::size_t j = 0; j < gap; j++)
                //{
                    u = arithmetic_.guard(x[j]);
                    v = arithmetic_.mul_root(y[j], r);
                    x[j] = arithmetic_.add(u, v);
                    y[j] = arithmetic_.sub(u, v);
                //}
            //}
        }

           
        void ntt_negacyclic_harvey_lazy(CoeffIter operand, const NTTTables &tables)
        {
            //this is the original function call
            // tables.ntt_handler().transform_to_rev(
            //     operand.ptr(), tables.coeff_count_power(), tables.get_from_root_powers());

            int log_n = tables.coeff_count_power();

            size_t n = size_t(1) << log_n;

           // std::uint64_t *values_device;
            //MultiplyUIntModOperand *roots_device;

            //cudaMalloc(&values_device, n * sizeof(std::uint64_t));
            //cudaMalloc(&roots_device, n * sizeof(MultiplyUIntModOperand));

            cudaMemcpy(values_device, operand.ptr(), n * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
            //cudaMemcpy(roots_device, tables.get_from_root_powers(), n * sizeof(MultiplyUIntModOperand), cudaMemcpyHostToDevice);

            std::size_t gap = n >> 1;
            std::size_t m = 1;

            //std::cout << "START" << std::endl;
            for (; m <= (n >> 8); m <<= 1)
            {
                //std::cout << "m: " << m << ", gap: " << gap << std::endl;
                transform_to_rev_kernel<<<gap, m>>>(tables.ntt_handler().arithmetic_, values_device, tables.get_from_root_powers_device(), gap, m);
                gap >>= 1;
            }
            for (; m <= (n >> 1); m <<= 1)
            {
                //std::cout << "m: " << m << ", gap: " << gap << std::endl;
                transform_to_rev_kernel2<<<m, gap>>>(tables.ntt_handler().arithmetic_, values_device, tables.get_from_root_powers_device(), gap, m);
                gap >>= 1;
            }
            //cudaDeviceSynchronize();

            cudaMemcpy(operand.ptr(), values_device, n * sizeof(std::uint64_t), cudaMemcpyDeviceToHost);
            //cudaFree(values_device);
            //cudaFree(roots_device);

        }

        void inverse_ntt_negacyclic_harvey_lazy(CoeffIter operand, const NTTTables &tables)
        {
            MultiplyUIntModOperand inv_degree_modulo = tables.inv_degree_modulo();
            tables.ntt_handler().transform_from_rev(
                operand.ptr(), tables.coeff_count_power(), tables.get_from_inv_root_powers(), &inv_degree_modulo);
        }

#ifdef __CUDA_ARCH__
        __host__ __device__
        void divide_uint128_uint64_inplace_generic(uint64_t *numerator, uint64_t denominator, uint64_t *quotient)
        {
// #ifdef SEAL_DEBUG
//             if (!numerator)
//             {
//                 throw invalid_argument("numerator");
//             }
//             if (denominator == 0)
//             {
//                 throw invalid_argument("denominator");
//             }
//             if (!quotient)
//             {
//                 throw invalid_argument("quotient");
//             }
//             if (numerator == quotient)
//             {
//                 throw invalid_argument("quotient cannot point to same value as numerator");
//             }
// #endif
            // We expect 128-bit input
            //renamed becasuse name conflict with function in modulus
            const size_t uint64_count_variable = 2;

            // Clear quotient. Set it to zero.
            quotient[0] = 0;
            quotient[1] = 0;

            // Determine significant bits in numerator and denominator.
            int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count_variable);
            int denominator_bits = get_significant_bit_count(denominator);

            // If numerator has fewer bits than denominator, then done.
            if (numerator_bits < denominator_bits)
            {
                return;
            }

            // Create temporary space to store mutable copy of denominator.
            uint64_t shifted_denominator[uint64_count_variable]{ denominator, 0 };

            // Create temporary space to store difference calculation.
            uint64_t difference[uint64_count_variable]{ 0, 0 };

            // Shift denominator to bring MSB in alignment with MSB of numerator.
            int denominator_shift = numerator_bits - denominator_bits;

            left_shift_uint128(shifted_denominator, denominator_shift, shifted_denominator);
            denominator_bits += denominator_shift;

            // Perform bit-wise division algorithm.
            int remaining_shifts = denominator_shift;
            while (numerator_bits == denominator_bits)
            {
                // NOTE: MSBs of numerator and denominator are aligned.

                // Even though MSB of numerator and denominator are aligned,
                // still possible numerator < shifted_denominator.
                if (sub_uint(numerator, shifted_denominator, uint64_count_variable, difference))
                {
                    // numerator < shifted_denominator and MSBs are aligned,
                    // so current quotient bit is zero and next one is definitely one.
                    if (remaining_shifts == 0)
                    {
                        // No shifts remain and numerator < denominator so done.
                        break;
                    }

                    // Effectively shift numerator left by 1 by instead adding
                    // numerator to difference (to prevent overflow in numerator).
                    add_uint(difference, numerator, uint64_count_variable, difference);

                    // Adjust quotient and remaining shifts as a result of shifting numerator.
                    quotient[1] = (quotient[1] << 1) | (quotient[0] >> (bits_per_uint64 - 1));
                    quotient[0] <<= 1;
                    remaining_shifts--;
                }
                // Difference is the new numerator with denominator subtracted.

                // Determine amount to shift numerator to bring MSB in alignment
                // with denominator.
                numerator_bits = get_significant_bit_count_uint(difference, uint64_count_variable);

                // Clip the maximum shift to determine only the integer
                // (as opposed to fractional) bits.
                int numerator_shift = min(denominator_bits - numerator_bits, remaining_shifts);

                // Shift and update numerator.
                // This may be faster; first set to zero and then update if needed

                // Difference is zero so no need to shift, just set to zero.
                numerator[0] = 0;
                numerator[1] = 0;

                if (numerator_bits > 0)
                {
                    left_shift_uint128(difference, numerator_shift, numerator);
                    numerator_bits += numerator_shift;
                }

                // Update quotient to reflect subtraction.
                quotient[0] |= 1;

                // Adjust quotient and remaining shifts as a result of shifting numerator.
                left_shift_uint128(quotient, numerator_shift, quotient);
                remaining_shifts -= numerator_shift;
            }

            // Correct numerator (which is also the remainder) for shifting of
            // denominator, unless it is just zero.
            if (numerator_bits > 0)
            {
                right_shift_uint128(numerator, denominator_shift, numerator);
            }
        }

#endif
        

    } // namespace util
} // namespace seal




 // __global__ void transform_from_rev_kernel(Arithmetic<std::uint64_t, MultiplyUIntModOperand, MultiplyUIntModOperand> arithmetic_,
 //                std::uint64_t *values, int log_n, const MultiplyUIntModOperand *roots, const MultiplyUIntModOperand *scalar)
 //            {
 //                // constant transform size
 //                size_t n = size_t(1) << log_n;
 //                // registers to hold temporary values
 //                MultiplyUIntModOperand r;
 //                std::uint64_t u;
 //                std::uint64_t v;
 //                // pointers for faster indexing
 //                std::uint64_t *x = nullptr;
 //                std::uint64_t *y = nullptr;
 //                // variables for indexing
 //                std::size_t gap = 1;
 //                std::size_t m = n >> 1;

 //                for (; m > 1; m >>= 1)
 //                {
 //                    std::size_t offset = 0;
 //                    if (gap < 4)
 //                    {
 //                        for (std::size_t i = 0; i < m; i++)
 //                        {
 //                            r = *++roots;
 //                            x = values + offset;
 //                            y = x + gap;
 //                            for (std::size_t j = 0; j < gap; j++)
 //                            {
 //                                u = *x;
 //                                v = *y;
 //                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
 //                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
 //                            }
 //                            offset += gap << 1;
 //                        }
 //                    }
 //                    else
 //                    {
 //                        for (std::size_t i = 0; i < m; i++)
 //                        {
 //                            r = *++roots;
 //                            x = values + offset;
 //                            y = x + gap;
 //                            for (std::size_t j = 0; j < gap; j += 4)
 //                            {
 //                                u = *x;
 //                                v = *y;
 //                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
 //                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

 //                                u = *x;
 //                                v = *y;
 //                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
 //                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

 //                                u = *x;
 //                                v = *y;
 //                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
 //                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

 //                                u = *x;
 //                                v = *y;
 //                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
 //                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
 //                            }
 //                            offset += gap << 1;
 //                        }
 //                    }
 //                    gap <<= 1;
 //                }

 //                if (scalar != nullptr)
 //                {
 //                    r = *++roots;
 //                    MultiplyUIntModOperand scaled_r = arithmetic_.mul_root_scalar(r, *scalar);
 //                    x = values;
 //                    y = x + gap;
 //                    if (gap < 4)
 //                    {
 //                        for (std::size_t j = 0; j < gap; j += 4)
 //                        {
 //                            u = arithmetic_.guard(*x);
 //                            v = *y;
 //                            *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
 //                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
 //                        }
 //                    }
 //                    else
 //                    {
 //                        for (std::size_t j = 0; j < gap; j += 4)
 //                        {
 //                            u = arithmetic_.guard(*x);
 //                            v = *y;
 //                            *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
 //                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);

 //                            u = arithmetic_.guard(*x);
 //                            v = *y;
 //                            *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
 //                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);

 //                            u = arithmetic_.guard(*x);
 //                            v = *y;
 //                            *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
 //                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);

 //                            u = arithmetic_.guard(*x);
 //                            v = *y;
 //                            *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
 //                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
 //                        }
 //                    }
 //                }
 //                else
 //                {
 //                    r = *++roots;
 //                    x = values;
 //                    y = x + gap;
 //                    if (gap < 4)
 //                    {
 //                        for (std::size_t j = 0; j < gap; j += 4)
 //                        {
 //                            u = *x;
 //                            v = *y;
 //                            *x++ = arithmetic_.guard(arithmetic_.add(u, v));
 //                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
 //                        }
 //                    }
 //                    else
 //                    {
 //                        for (std::size_t j = 0; j < gap; j += 4)
 //                        {
 //                            u = *x;
 //                            v = *y;
 //                            *x++ = arithmetic_.guard(arithmetic_.add(u, v));
 //                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

 //                            u = *x;
 //                            v = *y;
 //                            *x++ = arithmetic_.guard(arithmetic_.add(u, v));
 //                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

 //                            u = *x;
 //                            v = *y;
 //                            *x++ = arithmetic_.guard(arithmetic_.add(u, v));
 //                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

 //                            u = *x;
 //                            v = *y;
 //                            *x++ = arithmetic_.guard(arithmetic_.add(u, v));
 //                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
 //                        }
 //                    }
 //                }
 //            }