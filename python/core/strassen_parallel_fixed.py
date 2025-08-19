def _strassen_parallel(A11, A12, A21, A22, B11, B12, B21, B22):
    """Parallel execution with slab allocator and job-stealing"""
    n = A11.shape[0]
    
    # Prepare sub-problems using slab allocator
    tasks = []
    
    if PHYSICS_AVAILABLE:
        # Use slab pool for all temporaries
        with slab_pool.strassen_temps(n, count=7) as temps:
            # P1: (A11 + A22)(B11 + B22)
            np.add(A11, A22, out=temps[0])
            with slab_pool.get((n, n)) as temp_b:
                np.add(B11, B22, out=temp_b)
                tasks.append((hyperbolic_matrix_multiply, (temps[0].copy(), temp_b.copy()), {}))
            
            # P2: (A21 + A22)B11
            np.add(A21, A22, out=temps[1])
            tasks.append((hyperbolic_matrix_multiply, (temps[1].copy(), B11), {}))
            
            # P3: A11(B12 - B22)
            with slab_pool.get((n, n)) as temp_b:
                np.subtract(B12, B22, out=temp_b)
                tasks.append((hyperbolic_matrix_multiply, (A11, temp_b.copy()), {}))
            
            # P4: A22(B21 - B11)
            with slab_pool.get((n, n)) as temp_b:
                np.subtract(B21, B11, out=temp_b)
                tasks.append((hyperbolic_matrix_multiply, (A22, temp_b.copy()), {}))
            
            # P5: (A11 + A12)B22
            np.add(A11, A12, out=temps[2])
            tasks.append((hyperbolic_matrix_multiply, (temps[2].copy(), B22), {}))
            
            # P6: (A21 - A11)(B11 + B12)
            np.subtract(A21, A11, out=temps[3])
            with slab_pool.get((n, n)) as temp_b:
                np.add(B11, B12, out=temp_b)
                tasks.append((hyperbolic_matrix_multiply, (temps[3].copy(), temp_b.copy()), {}))
            
            # P7: (A12 - A22)(B21 + B22)
            np.subtract(A12, A22, out=temps[4])
            with slab_pool.get((n, n)) as temp_b:
                np.add(B21, B22, out=temp_b)
                tasks.append((hyperbolic_matrix_multiply, (temps[4].copy(), temp_b.copy()), {}))
        
        # Use job-stealing parallel execution
        products = threading_utils.strassen_parallel(tasks)
    else:
        # Fallback to ThreadPoolExecutor
        sub_problems = [
            (A11 + A22, B11 + B22),  # P1
            (A21 + A22, B11),        # P2
            (A11, B12 - B22),        # P3
            (A22, B21 - B11),        # P4
            (A11 + A12, B22),        # P5
            (A21 - A11, B11 + B12),  # P6
            (A12 - A22, B21 + B22),  # P7
        ]
        
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = [executor.submit(hyperbolic_matrix_multiply, A_sub, B_sub) 
                      for A_sub, B_sub in sub_problems]
            products = [f.result() for f in futures]
    
    P1, P2, P3, P4, P5, P6, P7 = products
    
    # Use JIT-compiled combination if available
    if PHYSICS_AVAILABLE:
        C11, C12, C21, C22 = strassen_combine_jit(P1, P2, P3, P4, P5, P6, P7)
    else:
        # Fallback
        C11 = np.empty_like(P1)
        C12 = np.empty_like(P1)
        C21 = np.empty_like(P1)
        C22 = np.empty_like(P1)
        
        np.add(P1, P4, out=C11)
        np.subtract(C11, P5, out=C11)
        np.add(C11, P7, out=C11)
        
        np.add(P3, P5, out=C12)
        
        np.add(P2, P4, out=C21)
        
        np.subtract(P1, P2, out=C22)
        np.add(C22, P3, out=C22)
        np.add(C22, P6, out=C22)
    
    return _assemble(C11, C12, C21, C22)
