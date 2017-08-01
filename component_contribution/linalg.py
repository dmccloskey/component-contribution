# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:21:14 2015

@author: noore
"""

import numpy as np
import scipy

class LINALG(object):

    @staticmethod
    def svd(A):

        # numpy.linalg.svd returns U, s, V such that
        # A = U * s * V

        # however, matlab and octave return U, S, V such that
        # V needs to be conjugate transposed when multiplied:
        # A = U * S * V.H

        # we would like to stick to the latter standard, so we return
        # the transposed V here (assuming it is real)

        U, s, V = scipy.linalg.svd(A, full_matrices=True)
        S = np.matrix(np.zeros(A.shape))
        np.fill_diagonal(S, s)
        U = np.matrix(U)
        V = np.matrix(V)
        return U, S, V.T

    @staticmethod
    def _zero_pad_S(S, cids_orig, cids_joined):
        """
            takes a stoichiometric matrix with a given list of IDs 'cids' and adds
            0-rows so that the list of IDs will be 'cids_joined'
        """
        if not set(cids_orig).issubset(cids_joined):
            raise Exception('The full list is missing some IDs in "cids"')

        full_S = np.zeros((len(cids_joined), S.shape[1]))
        for i, cid in enumerate(cids_orig):
            S_row = S[i, :]
            full_S[cids_joined.index(cid), :] = S_row

        return np.matrix(full_S)

    @staticmethod
    def _invert_project(A, eps=1e-10):
        n, m = A.shape
        U, S, V = LINALG.svd(A)
        inv_A = V * np.linalg.pinv(S) * U.T

        r = (S > eps).sum()
        P_R   = U[:, :r] * U[:, :r].T
        P_N   = U[:, r:] * U[:, r:].T

        return inv_A, r, P_R, P_N

    @staticmethod
    def _row_uniq(A):
        """
            A procedure usually performed before linear regression (i.e. solving Ax = y).
            If the matrix A contains repeating rows, it is advisable to combine
            all of them to one row, and the observed value corresponding to that
            row will be the average of the original observations.

            Input:
                A - a 2D NumPy array

            Returns:
                A_unique, P_row

                where A_unique has the same number of columns as A, but with
                unique rows.
                P_row is a matrix that can be used to map the original rows
                to the ones in A_unique (all values in P_row are 0 or 1).
        """
        # convert the rows of A into tuples so we can compare them
        A_tuples = [tuple(A[i,:].flat) for i in range(A.shape[0])]
        A_unique = list(sorted(set(A_tuples), reverse=True))

        # create the projection matrix that maps the rows in A to rows in
        # A_unique
        P_col = np.matrix(np.zeros((len(A_unique), len(A_tuples))))

        for j, tup in enumerate(A_tuples):
            # find the indices of the unique row in A_unique which correspond
            # to this original row in A (represented as 'tup')
            i = A_unique.index(tup)
            P_col[i, j] = 1

        return np.matrix(A_unique), P_col

    @staticmethod
    def _col_uniq(A):
        A_unique, P_col = LINALG._row_uniq(A.T)
        return A_unique.T, P_col.T
    
    # @staticmethod
    # def _invert_project(A, eps=1e-10, method='numpy'):
    #     '''
    #     alternative call to "_invert_project"
    #     with additional options to use octave, R, or numpy
    #     '''
    #     n, m = A.shape
    #     if method == 'octave':
    #         from oct2py import Oct2Py
    #         oc = Oct2Py()
    #         U, S, V = oc.svd(A)
    #         s = np.diag(S)
    #         U = np.matrix(U)
    #         V = np.matrix(V)
    #         r = sum(abs(s) > eps)
    #         inv_S = np.matrix(np.diag([1.0/s[i] for i in xrange(r)]))
    #         inv_A = V[:, :r] * inv_S * U[:, :r].T
    #         P_R   = U[:, :r] * U[:, :r].T
    #         P_N   = U[:, r:] * U[:, r:].T

    #         return inv_A, r, P_R, P_N
    #     elif method == 'numpy':
    #         # numpy.linalg.svd returns U, s, V_H such that
    #         # A = U * s * V_H
    #         # however, matlab and octave return U, S, V such that
    #         # V needs to be transposed when multiplied:
    #         # A = U * S * V.T
    #         U, s, V_H = np.linalg.svd(A, full_matrices=True)
    #         V = V_H.T
    #         r = sum(abs(s) > eps)
    #         inv_S = np.matrix(np.diag([1.0/s[i] for i in xrange(r)]))
    #         inv_A = V[:, :r] * inv_S * U[:, :r].T
    #         P_R   = U[:, :r] * U[:, :r].T
    #         P_N   = np.eye(n) - P_R

    #         return inv_A, r, P_R, P_N
    #     elif method == 'nosvd':
    #         inv_A = A.T * np.linalg.inv(A * A.T + np.eye(n)*4e-6).T
    #         # then the solution for (A.T * x = b) will be given by (x = inv_A.T * b)
    #         P_R = A * inv_A
    #         P_N = np.eye(n) - P_R
    #         r = sum(np.abs(np.linalg.eig(P_R)[0]) > 0.5)

    #         return inv_A, r, P_R, P_N
    #     elif method == 'r':
    #         # calculate average and CV of data
    #         # Call to R
    #         #base = importr('base');
    #         try:
    #             # convert matrix to R strings
    #             #A_rstring = (n*m-1)*(50*' ' + ',')
    #             #cnt = 0;
    #             #for i in range(n):
    #             #    for j in range(m):
    #             #        A_rstring = A_rstring[:cnt] + str(A[i,j]) + A_rstring[cnt+1:];
    #             #        cnt = cnt+50+1;
    #             A_list = []
    #             cnt = 0;
    #             for i in range(n):
    #                 for j in range(m):
    #                     A_list.append(A[i,j]),
    #             A_rstring = str(A_list)
    #             A_rstring = A_rstring[1:]
    #             A_rstring = A_rstring[:len(A_rstring)-1]
    #             r_statement = ('A = matrix(c(%s),nrow = %s, ncol = %s, byrow = TRUE' % (A_rstring,n,m))     
    #             ans = robjects.r(r_statement)
    #             ans = robjects.r('s = svd(A)')

    #             s = np.diag(ans.rx2['d'])
    #             U = np.matrix(ans.rx2['u'])
    #             V = np.matrix(ans.rx2['v'])
    #             r = sum(abs(s) > eps)

    #             inv_S = np.matrix(np.diag([1.0/s[i] for i in xrange(r)]))
    #             inv_A = V[:, :r] * inv_S * U[:, :r].T
    #             P_R   = U[:, :r] * U[:, :r].T
    #             P_N   = U[:, r:] * U[:, r:].T

    #             return inv_A, r, P_R, P_N
    #         except Exception as e:
    #             print(e)
    #     else:
    #         raise ArgumentError('method argument must be "octave", "numpy" or "nosvd"')
