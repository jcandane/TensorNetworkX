import torch

def su(N, dtype=torch.complex128): #JW=False, 
    """
    j.candanedo rev 8/11/2023
    GIVEN:  N , dimension of Lie-Algebra
            JW, choice for Jordan-Wigner Transformation
    GET:    sparse-array (t=array-of-tuples, T=data-array, out_shape=dense-array-shape)
    """

    if not dtype.is_complex:
        sparse_elements = int( N*(N-1) + N*(N+1)/2 - 1 )
        T = torch.zeros(sparse_elements, dtype=dtype)
    else:
        sparse_elements = int( 2*N*(N-1) + N*(N+1)/2 - 1 )
        T = torch.zeros(sparse_elements, dtype=dtype)
    i = 0 ## sparse-array entry
    j = 0 ## generator-number
    t = torch.zeros((sparse_elements,3), dtype=torch.int32)
    for n in range(N-1): ## loop over Cartan-generators
        for m in range(n+1): ## loop over off-diagonals
            ## real generator
            t[i] = torch.tensor([j, m, n+1])
            T[i] = 1.
            i+=1
            if not dtype.is_complex:
                j+=1
            t[i] = torch.tensor([j, n+1, m])
            T[i] = 1.
            i+=1
            j+=1

            ## imag generator
            if dtype.is_complex:
                t[i] = torch.tensor([j, m, n+1])
                T[i] = -1.j
                i+=1
                t[i] = torch.tensor([j, n+1, m])
                T[i] =  1.j
                i+=1
                j+=1

        ## constant for Cartan-genorators.
        C = 1/math.sqrt( (n+1)*(n+2)/2 ) ## constant for Cartan-genorators.
        for m in range(n+1): ## loop over sparse Cartan-generator elements
            ## place-in Cartan-elements
            t[i] = torch.tensor([j, m, m])
            T[i] = C
            i+=1
        ## last element of Cartan generator
        t[i] = torch.tensor([j, n+1, n+1])
        T[i] = -(n+1)*C
        i+=1
        j+=1
    out_shape = torch.tensor([j,N,N])
    return torch.sparse_coo_tensor(t.T, T, [int(i.item()) for i in out_shape])

def NN2_site(M, b):
    """ jcandanedo 4/2/23
    create the superoperator index-array
    GIVEN : M (size) [int]
            b (bitstring) [Boolean-np.array, int-np.array]
    GET   : O (superoperator index-array) [2D int-np.array, (dimension, entries), eg (7,9), 7=dimension]
    """
    bb=(M-1)*torch.as_tensor(b) ## the non-trivial-zero-location
    min,max=0,0
    O = bb[:,None]*torch.ones(len(b)*(M-1)+1, dtype=int)[None,:]
    for i in range(len(b)):
        max+=(M-1)
        if b[i]==0: ## if L
            O[i,min:(max)] = torch.arange(1,M)
        else: ## R
            O[i,min:(max)] = torch.arange(M-1)
        min=max
    return O

def get_tno_element(N, b, H1=None, dtype=torch.complex128): ### do H1
    """ jcandanedo 8/21=23, dtype=torch.float64 still under construction!!
    GIVEN:  N (int, SU(N) Lie-Algebra-order)
            b (bitstring boolean-List/torch.tensor)
            H1 (*optional a 2d-torch.tensor)
            dtype = (if complex to regular, if real/float do Jordan-Wigner, if int do JW with multipicitive constant)
    GET:    (torch.tensor{sparse})
    """

    t   = su(N, dtype=dtype)     ## interactions enumerated....
    M   = N**2-1+2  ## super-operator length
    TNO = NN2_site(M, b) #tno_structure(M, ell)
    tno_= [] ## index-array
    tnod= [] ## data

    for I in range(TNO.shape[1]-1): ## END M-1 from to negect nontrival 0 (xor 1-body Ham. here)
        ## tuple element under-consideration
        A = TNO[:,I]

        ## find corners and save identity-matrix
        cornerdetect = torch.logical_or( A==0, A==(M-1) ) ## corner-detector ## next locate special corner b = 
        if torch.all(cornerdetect): ## logical-AND
            special_corner = (M-1)*torch.as_tensor(b)
            if torch.all(cornerdetect==special_corner) and (H1 is not None):
                H1 = H1.to_sparse()
                tno_.append(H1._indices)
                tnod.append(H1._values)

            corner = torch.zeros((N,len(A)+2), dtype=int)
            corner[:,:len(A)]  = A[None,:]*torch.ones(N,dtype=int)[:,None]
            corner[:,len(A)]   = torch.arange(N)
            corner[:,len(A)+1] = torch.arange(N)
            tno_.append(corner)
            tnod.append(torch.ones(N))
        ## save Lie-algebra (find non 0,M element: 1,2,3,4,5,...,M-2,M-1)
        else:
            q    = TNO[ torch.logical_not(cornerdetect), I ][0]-1 ## search this
            L,R  = torch.searchsorted(t._indices()[0,:], q, side="left"), torch.searchsorted(t._indices()[0,:], q, side="right")
            outt = ( t._indices()[1:,:][:,L:R] ).T ## extract part of Lie-Algebra

            side = torch.zeros((len(A)+2,len(outt)), dtype=int)

            side[:len(A),:] = A[:,None]*torch.ones(len(outt),dtype=int)[None,:]
            side[len(A):,:] = outt.T
            tno_.append(side.T)
            tnod.append(t._values()[L:R]/2)

    indices = torch.concat(tno_).T
    data    = torch.concat(tnod)
    shapee  = torch.concat(( M*torch.ones(len(b),dtype=int), torch.tensor(t.shape[1:])))
    return torch.sparse_coo_tensor(indices, data, [int(i.item()) for i in shapee])