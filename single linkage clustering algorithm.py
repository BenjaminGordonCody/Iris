import math

class Clustering:
    """
    Takes a dataframe of x/y co-ordinates and generates a distance table for use in single linkage clustering. 
    The next_epoch method may then be used to work through each reclustering until all points in the dataframe have been combined in a single dendrogram.
    """

    def euclidean_distance(self, ax, ay, bx, by):
        ed = math.sqrt((abs(ax-bx) + abs(ay-by)))
        return ed

    def get_coord(self, point):
        
        x = self.points.at[point, "sepal_width"]
        y = self.points.at[point, "sepal_length"]

        return x,y

    def get_distance_matrix(self):
        
        # create empty 2d matrix
        rows, cols = (6, 6)
        dm = [[0 for i in range(cols)] for j in range(rows)]

        # fill in values 
        for i in range(rows):
            point_a = self.get_coord(i)
            for j in range(cols):

                #this if/else is to stop doubling of information between identical i:j and j:i
                if i <= j:
                    dm[i][j] = None
                    continue
                else:
                    point_b = self.get_coord(j)
                    dm[i][j] = self.euclidean_distance(*point_a, *point_b)
        
        # remove blank fields
        for i in dm:
            while None in i:
                i.remove(None)

        return dm
    
    def create_cluster_records(self):
        for i in range(len(self.points)):
            self.points.at[i, "cluster_record"] = i
    
    def __init__(self, dataframe):
        self.points = dataframe
        self.distance_matrix = self.get_distance_matrix()
        self.create_cluster_records()
        
    
    def next_cluster(self):
        
        #abbreviated here for more legible code
        dm = self.distance_matrix
        
        # this tuple, once filled with appropriate values, will be passed back to caller
        next_cluster = {
            "distance" : 99999, # an arbitrary high number 
            "i" : None,
            "j" : None,
        }
        
        # iterate over distance matrix
        for i in range(len(dm)):
            for j in range(len(dm[i])):
                
                # 'is this the shortest distance we've seen?
                if dm[i][j] > next_cluster["distance"]:
                    continue
                
                # 'are these two points in different clusters? '
                elif self.points.at[i, "cluster_record"] == self.points.at[j, "cluster_record"]:
                    continue
                
                # record of best candidate for clustering based on this pass of dm so far
                else:
                    next_cluster = {
                        "distance" : dm[i][j],
                        "i" : i,
                        "j" : j,
                    }
        return next_cluster
    
    def join_clusters(self, next_cluster):
        
        #abbreviation for easier reading
        df = self.points
        i = next_cluster["i"]
        j = next_cluster["j"]
        
        # what clusters do points i and j belong to?
        i = df.at[i, "cluster_record"]
        j = df.at[j, "cluster_record"]
        print(i,j)
        
        # join all of j's cluster to i's cluster
        self.points.replace({"cluster_record": j,}, i, inplace=True)
                
    def next_epoch(self):
        
        # abbreviations
        df = self.points
        
        # find candidates for clustering 
        next_cluster = self.next_cluster()
        print(next_cluster)
        print("\n")
        
        # make the cluster and print the cluster_record
        self.join_clusters(next_cluster)
        print(df.sort_values("cluster_record"))
        print("\n")
        
        # plot showing current clusters
        plt.show(
            sns.scatterplot(data=df, x="sepal_width", y="sepal_length", hue="cluster_record", palette="Set2")
        )
