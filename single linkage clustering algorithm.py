class app:
    '''this class takes a dataframe as an input and 
    def __init__(self, dataframe):
        self.points = dataframe

    def euclidean_distance(ax, ay, bx, by):
        ed = abs(ax-bx) + abs(ay-by)
        return ed

    def get_coord(self, point):
        
        x = None 
        y = None

        return x,y

    def get_distance_matrix(self):
        rows, cols = (6, 6)
        dm = [[0 for i in range(cols)] for j in range(rows)]

        for i in rows:
            point_a = get_coord(str(i))
            for j in cols:
                point_b = get_coord(str(i))
                dm[i][j] = euclidean_distance(*point_a, *point_b)

        self.distance_matrix = dm


