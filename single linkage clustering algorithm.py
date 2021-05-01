class app:
    
    def __init__(dataframe):
        self.points = dataframe

    def euclidean_distance(ax, ay, bx, by):
        ed = abs(ax-bx) + abs(ay-by)
        return ed

    def    
    
    def get_distance_matrix():
        rows, cols = (6, 6)
        dm = [[0 for i in range(cols)] for j in range(rows)]

        for i in rows:
            for j in cols:
                dm[i][j] = euclidean_distance(*point_a, *point_b)
        
        self.distance_matrix = dm c


