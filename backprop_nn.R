#SIGMOID
#takes array as an input and provides sigmoid as the output
sigmoid <- function(tvar) {
    s <- 1/(1+exp(-tvar))
    return(s)
}

#Initialize Parameters
initialize_with_zeros <- function(dim) {
    w = as.matrix(rep(0,dim), nrow = dim)
    b = 0
    params <- list("w" = w, "b" = b)
    return(params)
}

#forward propagation (dot product of weights with vector)
#X_mtx is my input matrix. Yt is my ground truth labels
#w = weights
#b = bias

propagate <- function(w, b, X_mtx, Yt) {
    #Forward propagation
    
    m = dim(X_mtx)[2] #total number of examples
    A = sigmoid((t(w) %*% X_mtx) + b)
    
    cost = -(1/m) * sum((Yt * log(A))+ ((1-Yt) * log(1-A)))
    
    
    #Backward Propagation
    dw = (1 / m) * (X_mtx %*% t(A - Yt))
    db = (1 / m) * sum(A - Yt)
    grads <- list("dw" = dw, "db" = db, "cost" = cost)
    return(grads)
}

optimize <- function(w, b, X_mtx, Yt, num_iterations, learning_rate) {
    costs = c()
    for (i in 1:num_iterations){
        grads <- propagate(w, b, X_mtx, Yt)
        dw <- grads$dw
        db <- grads$db
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if (i %% 10 == 0){
            costs <- c(costs,grads$cost)
        }
        
        if (i %% 10 == 0){
            print (paste0("Cost after iteration ", i, " is ", grads$cost))
        }
        
    }
    params <- list('w' = w, 'b' = b, 'dw' = dw, 'db' = db, 'cost' = costs)
    
    return(params)
}


input_params <- initialize_with_zeros(3)
w <- input_params$w
b <- input_params$b

optimize(w, b, X_mtx, Yt, num_iterations = 1000, learning_rate = .09)
#Test Case
a=matrix(c(1,2,3),ncol=3)
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
