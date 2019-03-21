# Step 1: Characterizing the pathloss
m, c = np.linalg.lstsq(
     np.vstack([logdistance,
        P_0 * np.ones(len(logdistance))
        ]).T,
     prx,
     rcond=None)[0]

# The intercept should be equal to P_0, which is certainly
# guaranteed by the linear system. Adding an assert in order
# to make it explicit.
# XXX: Note it may fail because of float point precision, it
# is not an issue for P_0 = 0 though.
assert c == P_0
