/* TODO */
#include <causality.h>
#include <dataframe.h>
#include <scores.h>
#include <R_ext/Lapack.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define LOOP_UNROLL_SIZE 8

#define ERROR_THRESH     1e-6
#define NOT_POSITIVE_DEFINITE 1
#define NON_POSITIVE_RESIDUAL_VARIANCE 2

double calculate_rss(double *cov_xy, double *cov_xx, int npar);
double fddot(double *x, double *y, int n);
void   fcov_xx(double *cov_xx, double **df, int npar, int nobs);
void   fcov_xy(double *cov_xy, double **df, double *y, int npar, int nobs);

/* TODO */
double bic_score(struct dataframe data, int x, int y, int *ypar, int npar,
                                        double *fargs, int *iargs)
{
    double **_x = malloc((npar + 1) * sizeof(double));
    _x[0] = data.df[x];
    for (int i = 0; i < npar; ++i)
        _x[i + 1] = data.df[ypar[i]];
    double *_y = data.df[y];
    double penalty = fargs[0];
    /* Allocate memory for cov_xx and cov_xy in one block. */
    double *plus_mem = malloc((npar + 1) * (npar + 2) * sizeof(double));
    if (plus_mem == NULL)
        error("failed to allocate memory for BIC score\n");
    double *cov_xx_plus     = plus_mem;
    double *cov_xy_plus     = plus_mem + (npar + 1) * (npar + 1);
    double *minus_mem = malloc(npar  * (npar + 1) * sizeof(double));
    if (minus_mem == NULL)
        error("failed to allocate memory for BIC score\n");
    double *cov_xx_minus     = minus_mem;
    double *cov_xy_minus     = minus_mem + npar * npar;

    /* Calculate the covariance matrix of the data matrix of x */
    fcov_xx(cov_xx_plus, _x, npar + 1, data.nobs);
    /* calculate the covariance vector between y and x */
    fcov_xy(cov_xy_plus, _x, _y, npar + 1, data.nobs);



    for (int i = 0; i < npar; ++i)
        cov_xy_minus[i] = cov_xy_plus[i + 1];
    for (int j = 0; j < npar; ++j) {
        for (int i = 0; i < npar; ++i)
            cov_xx_minus[i + j*npar] = cov_xx_plus[(i + 1) + (npar + 1) * (j + 1)];
    }
    double rss_plus  = calculate_rss(cov_xy_plus, cov_xx_plus, npar + 1);
    double bic_plus  = data.nobs * log(rss_plus) +
                         penalty * log(data.nobs) * (2 * (npar + 1) + 1);
    double rss_minus = calculate_rss(cov_xy_minus, cov_xx_minus, npar);
    double bic_minus = data.nobs * log(rss_minus) +
                         penalty * log(data.nobs) * (2 * npar + 1);
    free(plus_mem);
    free(minus_mem);
    return bic_plus - bic_minus;
}


/*
 * Now, we shall calcluate the BIC rss of this configuration by computing
 * log(cov_xx - cov_xy**T cov_xx^-1 * cov_xy) + log(n) * (npar + 1). The
 * first (and main step) in the rest of this function is to calcluate
 * cov_xy**T cov_xx^-1 * cov_xy.  Note that because we all variables are
 * normalized variables, cov_yy = 1
 */
double calculate_rss(double *cov_xy, double *cov_xx, int npar)
{
    int    err = 0;
    double rss = 1.0f;
    if (npar == 0)
        return rss;
    if (npar == 1) {
        rss -= cov_xy[0] * cov_xy[0];
    }
    /*
     * In the 2x2 case, we have cov_xx = |1, cov(x1, x2)|
     *                                   |cov(x2, x1), 1|
     */
    else if (npar == 2) {
        double det = 1.0f - cov_xx[1] * cov_xx[1];
        /*
         * For a 2x2 symmetric matrix with 1's on the diagonal, having a non
         * positive determinent is enough to know that the matrix is not
         * positive definite
         */
        if (det < ERROR_THRESH)
            err = NOT_POSITIVE_DEFINITE;
        else
            /* formula by hand */
            rss -= (cov_xy[0] * cov_xy[0] + cov_xy[1] * cov_xy[1]
                   - 2 * cov_xx[1] * cov_xy[0] * cov_xy[1])/det;
        }
    /*
     * since npar > 2, we will now use a few LAPACK routines to solve the
     * equation cov_xy**T * cov_xx^-1 * cov_xy via the cholesky decomposition
     * instead of doing it by hand.
     */
    else {
        int lapack_err = 0;
        /*
         * We need to calculate the Cholesky decomposition of cov_xx so we can
         * use it to solve the linear system cov_xx * X = cov_xy. Thus, we use
         * the LAPACK subroutine dpotrf to achieve this.
         */
        F77_CALL(dpotrf)("L",
                         &npar,      /* nrow/ncols of cov_xx */
                         cov_xx,     /* input */
                         &npar,      /* stride of cov_xx */
                         &lapack_err /* we use this to check for errors */
        );
        /* Check to see if cov_xx is not positive definite */
        if (lapack_err) {
            err = NOT_POSITIVE_DEFINITE;
            goto END; /* We can skip the next LAPACK routine */
        }
        /* We need to create a copy of cov_xy to perform the next subroutine */
        double *cov_xy_cpy = malloc(npar * sizeof(double));
        memcpy(cov_xy_cpy, cov_xy, npar * sizeof(double));
        /* Now, we use the LAPACK routine dpotrs to solve the aforemention system
        * cov_xx * X = cov_xy. cov_xy_cpy is modified in place to be transformed
        * into X, which is why we created a copy of cov_xy. Note that, assuming
        * cov_xx is positive definite, X = cov_xx^-1 * cov_xy */
        int one = 1;
        F77_CALL(dpotrs)("L",
                         &npar, &one,
                         cov_xx,
                         &npar,
                         cov_xy_cpy, /* modified in place */
                         &npar,
                         &lapack_err
        );
        if (lapack_err)
            err = NOT_POSITIVE_DEFINITE;
        else
            rss -= fddot(cov_xy, cov_xy_cpy, npar);
        END: {};
    }
    if (rss < ERROR_THRESH) {
        err = NON_POSITIVE_RESIDUAL_VARIANCE;
    }

    if (err) {
        warning("The augmented matrix xy is not of full rank!\n");
        return DBL_MAX;
    }
    return rss;
}

/*
 * fcov_xx in theory should provide a fast calculation of covariance matrix
 * of the ranodom vector y. It attempts to store intermediate rsss as much as
 * possible, pull constants for the loop out of loop, and inludes a loop
 * unrolling type technique that will directly caclulate small (dim(y) <= 2)
 * covariance matrices instead of going through the loop, which is much slower.
 * Profiling might add more loop unrolling. Furthermore, this acts directly on R
 * data.frames, so we need to calculate cov_xx anyway to create a compact matrix
 * libRblas can operate on.
 */
void fcov_xx(double *cov_xx, double **x, int npar, int nobs)
{
    double inv_nminus1 = 1.0f/(nobs - 1.0f);
    for (int j = 0; j < npar; ++j) {
        double *x_j          = x[j];
        double *cov_xx_jnp = cov_xx + j * npar;
        double *cov_xx_jof = cov_xx + j;
        for (int i = 0; i <= j; ++i) {
            /* cov(x_i, x_i) == 1 */
            if (i == j)
                cov_xx_jnp[i]     = 1.0f;
            else
                cov_xx_jof[i*npar] =
                cov_xx_jnp[i]      = fddot(x_j, x[i], nobs) * inv_nminus1;
        }
    }
}

/*
 * fcov_xy caclulates the covariance vector between the ranodom variable x
 * and the random variable vector, y.
 */
void fcov_xy(double *cov_xy, double **x, double *y, int npar, int nobs)
{
    double inv_nminus1 = 1.0f/(nobs - 1.0f);
    for (int i = 0 ; i < npar; ++i)
        cov_xy[i] = fddot(x[i], y, nobs) * inv_nminus1;
}

/* fddot should do reasonably fast dot products by employing loop unrolling */
double fddot(double * restrict x, double * restrict y, int n)
{
    int q = n / LOOP_UNROLL_SIZE;
    int r = n % LOOP_UNROLL_SIZE;
    double psums[LOOP_UNROLL_SIZE] = {0.0f};
    for (int i = 0; i < q; ++i) {
        int i_lus = i * LOOP_UNROLL_SIZE;
        psums[0] += x[i_lus + 0] * y[i_lus + 0];
        psums[1] += x[i_lus + 1] * y[i_lus + 1];
        psums[2] += x[i_lus + 2] * y[i_lus + 2];
        psums[3] += x[i_lus + 3] * y[i_lus + 3];
        psums[4] += x[i_lus + 4] * y[i_lus + 4];
        psums[5] += x[i_lus + 5] * y[i_lus + 5];
        psums[6] += x[i_lus + 6] * y[i_lus + 6];
        psums[7] += x[i_lus + 7] * y[i_lus + 7];
    }
    q = q * LOOP_UNROLL_SIZE;
    switch (r) {
    case 7: psums[6] += x[q + 6] * y[q + 6];
    case 6: psums[5] += x[q + 5] * y[q + 5];
    case 5: psums[4] += x[q + 4] * y[q + 4];
    case 4: psums[3] += x[q + 3] * y[q + 3];
    case 3: psums[2] += x[q + 2] * y[q + 2];
    case 2: psums[1] += x[q + 1] * y[q + 1];
    case 1: psums[0] += x[q + 0] * y[q + 0];
    case 0: ;
    }
    return psums[0] + psums[1] + psums[2] + psums[3] + psums[4] + psums[5] +
            psums[6] + psums[7];

}
