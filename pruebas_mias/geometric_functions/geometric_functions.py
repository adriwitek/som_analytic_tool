


'''
    Geometric functions from minisom


'''
    
    
    
    def asymptotic_decay(learning_rate, t, max_iter):
        """Decay function of the learning process.
        Parameters
        ----------
        learning_rate : float
            current learning rate.
        t : int
            current iteration.
        max_iter : int
            maximum number of iterations for the training.
        """
        return learning_rate / (1+t/(max_iter/2))



    #Funciones de Vecindad
    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2*pi*sigma*sigma
        ax = exp(-power(self._xx-self._xx.T[c], 2)/d)
        ay = exp(-power(self._yy-self._yy.T[c], 2)/d)
        return (ax * ay).T  # the external product gives a matrix

    def _mexican_hat(self, c, sigma):
        """Mexican hat centered in c."""
        p = power(self._xx-self._xx.T[c], 2) + power(self._yy-self._yy.T[c], 2)
        d = 2*pi*sigma*sigma
        return (exp(-p/d)*(1-2/d*p)).T

    def _bubble(self, c, sigma):
        """Constant function centered in c with spread sigma.
        sigma should be an odd value.
        """
        ax = logical_and(self._neigx > c[0]-sigma,
                         self._neigx < c[0]+sigma)
        ay = logical_and(self._neigy > c[1]-sigma,
                         self._neigy < c[1]+sigma)
        return outer(ax, ay)*1.

    def _triangle(self, c, sigma):
        """Triangular function centered in c with spread sigma."""
        triangle_x = (-abs(c[0] - self._neigx)) + sigma
        triangle_y = (-abs(c[1] - self._neigy)) + sigma
        triangle_x[triangle_x < 0] = 0.
        triangle_y[triangle_y < 0] = 0.
        return outer(triangle_x, triangle_y)

    #####

    def _cosine_distance(self, x, w):
        num = (w * x).sum(axis=2)
        denum = multiply(linalg.norm(w, axis=2), linalg.norm(x))
        return 1 - num / (denum+1e-8)

    def _euclidean_distance(self, x, w):
        return linalg.norm(subtract(x, w), axis=-1)

    def _manhattan_distance(self, x, w):
        return linalg.norm(subtract(x, w), ord=1, axis=-1)

    def _chebyshev_distance(self, x, w):
        return max(subtract(x, w), axis=-1)