#from sage.all import *
import numpy as np
import random, networkx as nx

def kernPause(a1,a2):
    return  1*(a1==a2)

PREFIX_FOLDER = "/home/musescore1983/flask_apps/bezier_music/"

def kernPause(a1,a2):
    return  1*(a1==a2)

def kernDuration(d1,d2):
    return min(d1,d2)/max(d1,d2)

def kernVolume(v1,v2):
    return min(v1,v2)/max(v1,v2)

def getlowestfraction(x0):
    eps = 0.01

    x = np.abs(x0)
    a = np.floor(x)
    h1 = 1
    k1 = 0
    h = a
    k = 1

    while np.abs(x0-h/k)/np.abs(x0)> eps:
        x = 1/(x-a)
        a = np.floor(x)
        h2 = h1
        h1 = h
        k2 = k1
        k1 = k
        h = h2 + a*h1
        k = k2 + a*k1
    q = {"numerator": h, "denominator" :k}
    return q



def getRational(k):
    x = 2**(k*(1/12.0))
    return getlowestfraction(x)

def gcd(a,b):
    a = abs(a)
    b = abs(b)
    if (b > a):
        temp = a
        a = b
        b = temp
    while True:
        if (b == 0):
            return a
        a %= b
        if (a == 0):
            return b
        b %= a



def kernPitch(k1,k2):
    q = getRational(k2-k1)
    a,b = q["numerator"],q["denominator"]
    return gcd(a,b)**2/(a*b)

def kernNote(n1,n2):
    p1,d1,v1,r1 = n1
    p2,d2,v2,r2 = n2
    #print(n1,n2)
    return 1.0/(1+2+4+8)*(2*kernPitch(p1,p2)+4*min(d1,d2)/max(d1,d2)+1*min(v1,v2)/max(v1,v2)+8*kernPause(r1,r2))

import portion as PP

def muInterval(i):
    if i.empty:
        return 0
    return i.upper-i.lower

def jaccard(i1,i2):
    return muInterval(i1.intersection(i2))/muInterval(i1.union(i2))

def intersectionKernel(i1,i2):
    return muInterval(i1.intersection(i2))

def kernJacc(interval1,interval2):
    min1,max1 = interval1
    min2,max2 = interval2
    eps = 1.0/2048.0
    X = PP.closed(min1+eps,max1-eps)
    Y = PP.closed(min2+eps,max2-eps)
    return jaccard(X,Y)

def kernChord(c1,c2):
    n1 = len(c1)
    n2 = len(c2)
    s = 0
    #print(c1)
    return 1.0/(n1*n2)*np.sum([ kernNote(n1,n2) for n1 in c1 for n2 in c2])

def kernInt(i1,i2):
    voice1,min1, max1,note1 = i1 
    voice2,min2, max2,note2 = i2
    return kernChord(note1,note2)*jaccard(PP.closed(min1,max1),PP.closed(min2,max2))
    #return kernChord(note1,note2)*intersectionKernel(PP.closed(min1,max1),PP.closed(min2,max2))

def kernIntChord(i1,i2):    
    voice1,min1, max1,note1 = i1 
    voice2,min2, max2,note2 = i2
    return kernChord(note1,note2)
    
def kernConnectedComponent(kk=kernInt):
    return (lambda c1,c2: 1.0/(len(c1)*len(c2))*np.sum([kk(i1,i2) for i1 in c1 for i2 in c2]))

def kernVolume(v1,v2):
    #return kernJacc(v1,v2)
    return min(v1,v2) #/max(v1,v2)

def kernDuration(d1,d2):
    return min(d1,d2)

def create_random_graph(n):
    G = Graph(loops=False)
    G.add_vertex(1-1)
    for k in range(2,n+1):
        vert = [v for v in G.vertices()]
        G.add_vertex(k-1)
        for v in vert:
            prob = 1.0/v*k/sigma(k)
            p = randint(1,100)/100.0
            #print(prob,p)
            if p <= prob and k%(v+1)==0:
                G.add_edge(v,k-1)
    return G

def create_divisor_graph(n):
    G = DiGraph(loops=True)
    divs = divisors(n)
    for d in divs:
        G.add_vertex(divs.index(d))
    for u in divs:
        for v in divs:
            if is_prime(u%v):
                G.add_edge(divs.index(u),divs.index(v))
    return G

def kernAdd(t1,t2,alphaPitch=0.25):
    pitch1,volume1,isPause1 = t1
    pitch2,volume2,isPause2 = t2
    #return 1.0/3*(1-alphaPitch)*kernPause(isPause1,isPause2)+alphaPitch*kernPitch(pitch1,pitch2)+1.0/3*(1-alphaPitch)*kernDuration(duration1,duration2)+1.0/3*(1-alphaPitch)*kernVolume(volume1,volume2)
    apa = alphaPitch["pause"]
    api = alphaPitch["pitch"]
    avo = alphaPitch["volume"]
    #return kernPause(isPause1,isPause2)*kernPitch(pitch1,pitch2)*kernVolume(volume1,volume2)

    if np.abs(apa+api+avo-1)<10**-5:
        return apa*kernPause(isPause1,isPause2)+api*kernPitch(pitch1,pitch2)+avo*kernVolume(volume1,volume2)
    else:
        return None

def kern0(zz0,alphaPitch={"pitch":1,"volume":2,"pause":3}):
    return lambda t1,t2: kernAdd(zz0[int(t1[0])],zz0[int(t2[0])],alphaPitch)

def kern(alphaPitch={"pitch":1,"volume":2,"pause":3}):
    return lambda t1,t2: kernAdd(t1,t2,alphaPitch)



def distKern1(x,y,alphaPitch={"pitch":1,"volume":2,"pause":3}):
    #print(alphaPitch)
    return np.sqrt(2-2*kern(alphaPitch)(x,y))

def distKern(kern):
    return lambda a,b : np.sqrt(kern(a,a)+kern(b,b)-2*kern(a,b))

import music21 as m21
from itertools import product

durlist = [[sum([((2**(n-i))) for i in range(d+1)]) for n in range(-8,3+1)] for d in range(2)]
durationslist = []
for dl in durlist:
    durationslist.extend([x for x in dl])
print(durationslist)

def findNearestDuration(duration,durationslist):
    return sorted([(abs(duration-nv),nv) for nv in durationslist])[0][1]

def parse_file_0(xml):
    import copy
    xml_data = m21.converter.parse(xml)
    score = []
    for part in xml_data.parts:
        parts = []
        for note in part.recurse().notesAndRests:
            if note.isRest:
                start = note.offset
                duration = float(note.quarterLength)/4.0
                vol = 32 #note.volume.velocity
                pitch = 60
                parts.append((copy.deepcopy(note),[pitch,findNearestDuration(duration,durationslist),vol,True]))
            elif note.isChord:
                note = [n for n in note][1]
                start = note.offset
                duration = float(note.quarterLength)/4.0
                pitch = note.pitch.midi
                #print(pitch,duration,note.volume)
                vol = note.volume.velocity
                if vol is None:
                    vol = int(note.volume.realized * 127)
                parts.append((copy.deepcopy(note),[pitch,findNearestDuration(duration,durationslist),vol,False]))    
            else:
                #print(note)
                start = note.offset
                duration = float(note.quarterLength)/4.0
                pitch = note.pitch.midi
                #print(pitch,duration,note.volume)
                vol = note.volume.velocity
                if vol is None:
                    vol = int(note.volume.realized * 127)
                parts.append((copy.deepcopy(note),[pitch,findNearestDuration(duration,durationslist),vol,False]))    
        score.append(parts)        
    return score


def writePitches(fn,inds,tempo=82,instrument=[0,0],add21=True,start_at= [0,0],durationsInQuarterNotes=False):
    from MidiFile import MIDIFile

    track    = 0
    channel  = 0
    time     = 0   # In beats
    duration = 1   # In beats # In BPM
    volume   = 116 # 0-127, as per the MIDI standard

    ni = len(inds)
    MyMIDI = MIDIFile(ni,adjust_origin=False) # One track, defaults to format 1 (tempo track
                     # automatically created)
    MyMIDI.addTempo(track,time, tempo)


    for k in range(ni):
        MyMIDI.addProgramChange(k,k,0,instrument[k])


    times = start_at
    for k in range(len(inds)):
        channel = k
        track = k
        for i in range(len(inds[k])):
            pitch,duration,volume,isPause = inds[k][i]
            #print(pitch,duration,volume,isPause)
            track = k
            channel = k
            if not durationsInQuarterNotes:
                duration = 4*duration#*maxDurations[k] #findNearestDuration(duration*12*4)            
            #print(k,pitch,times[k],duration,100)
            if not isPause: #rest
                #print(volumes[i])
                # because of median:
                pitch = int(floor(pitch))
                if add21:
                    pitch += 21
                #print(pitch,times[k],duration,volume,isPause)    
                MyMIDI.addNote(track, channel, int(pitch), float(times[k]) , float(duration), int(volume))
                times[k] += duration*1.0  
            else:
                times[k] += duration*1.0
       
    with open(fn, "wb") as output_file:
        MyMIDI.writeFile(output_file)
    print("written")  


def run_length_row(row_of_01):
    from itertools import groupby
    ar = row_of_01
    return [(k, sum(1 for i in g)) for k,g in groupby(ar)]

def int_comp_row(row_of_01,by=8):
    ll = divide_row_by(row_of_01,by=by)
    ss = []
    for l in ll:
        rl = run_length_row(l)
        ss.append([v for k,v in rl])
    return list(ss)    

def divide_row_by(row,by):
    ll = []
    n = len(row)
    m = n//by
    #print(m)
    for k in range(m):
        ll.append(row[k*by:((k+1)*by)])
    return list(ll)    

def generateNotes(pitchlist,alphaPitch={"pitch":1,"volume":2,"pause":3},shuffle_notes=True):
    from itertools import product
    from music21 import pitch
    #pitchlist = [p for p in list(range(60-1*octave*12,60+24-1*octave*12))]
    #distmat = np.array(matrix([[np.sqrt(2*(1.0-kernPitch(x,y))) for x in pitchlist] for y in pitchlist]))
    #permutation,distance = tspWithDistanceMatrix(distmat,exact=False)
    #pitchlist = [pitchlist[permutation[k]] for k in range(len(pitchlist))]
    print([pitch.Pitch(midi=int(p)) for p in pitchlist])
    #durationlist = [n for n in durs]
    #if len(durs)>2:
    #    distmat = np.array(matrix([[np.sqrt(2*(1.0-kernDuration(x,y))) for x in durationlist] for y in durationlist]))
    #    permutation,distance = tspWithDistanceMatrix(distmat)
    #    durationlist = [durationlist[permutation[k]] for k in range(len(durationlist))]
    #print(durationlist)
    volumelist = vols = [(128//8)*(k+1) for k in range(8)] #[x*127 for x in [1.0/6.0,1.0/3.0,1.0/2.0,2.0/3.0 ]]
    print(volumelist)
    #distmat = np.array(matrix([[np.sqrt(2*(1.0-kernVolume(x,y))) for x in volumelist] for y in volumelist]))
    #permutation,distance = tspWithDistanceMatrix(distmat)
    #volumelist = [volumelist[permutation[k]] for k in range(len(volumelist))]
    print(volumelist)
    pauselist = [False,True]
    ll = list(product(pauselist,volumelist,pitchlist))
    if shuffle_notes:
        shuffle(ll)
    #distmat = np.array(matrix([[distKern(x,y,alphaPitch) for x in ll] for y in ll]))
    #np.random.seed(43)
    #permutation,distance = tspWithDistanceMatrix(distmat,exact=False)
    #ll = [ll[permutation[k]] for k in range(len(ll))]
    print(len(ll))
    #print(ll)
    pitches = [p[2] for p in ll]
    #durations = [d[0] for d in ll]
    volumes = [v[1] for v in ll]
    isPauses = [p[0] for p in ll]
    #print(pitches)
    return pitchlist,volumelist,pauselist


def get_knn_model(for_list,kern=kernPitch):
    #notes = np.array([[x*1.0 for x in n] for n in notes])
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)
    #nbrs = NearestNeighbors( algorithm='ball_tree',metric=distKern(kern(zz0,alphaPitch=alphaPitch))).fit([[r] for r in range(len(zz0))])
    M = matrix([[float(kern(a,b)) for a in for_list] for b in for_list],ring=RDF)
    #print(M)
    Ch = np.array(M.cholesky())
    nbrs = NearestNeighbors().fit(Ch)
    return nbrs, Ch

def findBestMatches(nbrs,new_row,n_neighbors=3):
    distances,indices = nbrs.kneighbors([np.array(new_row)],n_neighbors=n_neighbors)
    dx = sorted(list(zip(distances[0],indices[0])))
    #print(dx)
    indi = [d[1] for d in dx]
    #print(indi)
    #print(distances)
    #distances,indices = nbrs.query([np.array(new_row)],k=n_neighbors)
    return indi


from itertools import groupby

def get_flattened_durations_from_graph(G,permutation_of_vertices,sorted_reverse,by=16,shuffled=True):
    zz = sorted(list(zip(G.degree_sequence(),G.vertices())),reverse=sorted_reverse)
    #print(sorted(zz))
    if shuffled:
        shuffle(zz)
    A = G.adjacency_matrix(vertices = permutation_of_vertices)
    ll = []
    for row in A:
        #print(row)
        ll.extend([xx/by for xx in x] for x in (int_comp_row(row,by=by)))
    ss = []
    for l in ll:
        ss.extend(l)
    return ss    

def get_bitstring_from_durations(durs,denom=None):
    # convert reciprocal into rational number:
    # take the denominator
    # take the gcd of all denominators
    qq = ([QQ(d) for d in durs])
    dd = [q.denominator() for q in qq]
    D = denom if not denom is None else lcm(dd)
    nn = [q.numerator()*D/q.denominator() for q  in qq]
    xx = []

    print(D,nn)
    y = 0
    for n in nn:
        xx.extend(n*[y])
        y = 1*(not y)
    return xx,D    
    
 
def get_durations_from_graph(G,sorted_reverse,by=16,shuffled=True):
    zz = sorted(list(zip(G.degree_sequence(),G.vertices())),reverse=sorted_reverse)
    #print(sorted(zz))
    if shuffled:
        shuffle(zz)
    A = G.adjacency_matrix(vertices = [z[1] for z in zz])
    ss = []
    for row in A:
        print(row)
        xx = ((int_comp_row(row,by=by)))
        ll = []
        for x in xx: 
            ll.extend([xs/by for xs in x])
        ss.append(ll)  
    return ss    


def interval_graph_slow(scores,eps=1.0/16.0):
    """Generates an interval graph for a list of intervals given.

    In graph theory, an interval graph is an undirected graph formed from a set
    of closed intervals on the real line, with a vertex for each interval
    and an edge between vertices whose intervals intersect.
    It is the intersection graph of the intervals.

    More information can be found at:
    https://en.wikipedia.org/wiki/Interval_graph

    Parameters
    ----------
    intervals : a sequence of intervals, say (l, r) where l is the left end,
    and r is the right end of the closed interval.

    Returns
    -------
    G : networkx graph

    Examples
    --------
    >>> intervals = [(-2, 3), [1, 4], (2, 3), (4, 6)]
    >>> G = nx.interval_graph(intervals)
    >>> sorted(G.edges)
    [((-2, 3), (1, 4)), ((-2, 3), (2, 3)), ((1, 4), (2, 3)), ((1, 4), (4, 6))]

    Raises
    ------
    :exc:`TypeError`
        if `intervals` contains None or an element which is not
        collections.abc.Sequence or not a length of 2.
    :exc:`ValueError`
        if `intervals` contains an interval such that min1 > max1
        where min1,max1 = interval
    """
    graph = nx.Graph()

    ints = []
    ddintm21note = dict([])
    for sci in range(len(scores)):
        sc = scores[sci]
        t = 0
        for note in sc[0:-1]:
            noteM21,nv = note
            pitch,duration,volume,isPause = nv[0]
            #if not isPause:
            #print(noteM21,nv)
            interval = (sci,t,t+duration,tuple(nv))
            graph.add_edge(interval,interval) #every interval is connected to itself
            if sci>0:
                con_comp = list(nx.connected_components(graph))
                print(len(con_comp))
                for cc in con_comp:
                    for n in cc:
                        if kernInt(interval,n)>eps:
                            graph.add_edge(interval,n)
            ints.append(interval)
            ddintm21note[interval] = noteM21
                #dintpitch[(t,t+duration)] = note
            t += duration    
    return graph, ddintm21note        
    

def interval_graph_0(scores,eps=None):
    """Generates an interval graph for a list of intervals given.

    In graph theory, an interval graph is an undirected graph formed from a set
    of closed intervals on the real line, with a vertex for each interval
    and an edge between vertices whose intervals intersect.
    It is the intersection graph of the intervals.

    More information can be found at:
    https://en.wikipedia.org/wiki/Interval_graph

    Parameters
    ----------
    intervals : a sequence of intervals, say (l, r) where l is the left end,
    and r is the right end of the closed interval.

    Returns
    -------
    G : networkx graph

    Examples
    --------
    >>> intervals = [(-2, 3), [1, 4], (2, 3), (4, 6)]
    >>> G = nx.interval_graph(intervals)
    >>> sorted(G.edges)
    [((-2, 3), (1, 4)), ((-2, 3), (2, 3)), ((1, 4), (2, 3)), ((1, 4), (4, 6))]

    Raises
    ------
    :exc:`TypeError`
        if `intervals` contains None or an element which is not
        collections.abc.Sequence or not a length of 2.
    :exc:`ValueError`
        if `intervals` contains an interval such that min1 > max1
        where min1,max1 = interval
    """

    ints = []
    ddintm21note = dict([])
    print("reading intervals from scores")
    for sci in range(len(scores)):
        sc = scores[sci]
        t = 0
        print("voice = ", sci)
        for note in sc[0:-1]:
            noteM21,nv = note
            pitch,duration,volume,isPause = nv[0]
            #if not isPause:
            #print(noteM21,nv)
            interval = (t,t+duration,(sci,tuple(nv)))
            #print(interval)
            ints.append(interval)
            ddintm21note[(sci,t,t+duration,tuple(nv))] = noteM21
                #dintpitch[(t,t+duration)] = note
            t += duration          
    graph = nx.Graph()
    from intervaltree import Interval
    
    tupled_intervals = [tuple(interval) for interval in ints]
    maxTime = int(np.ceil(max([i[1] for i in tupled_intervals])))
    minDiff  = min([i[1]-i[0] for i in tupled_intervals])
    print(maxTime,minDiff)
    #graph.add_nodes_from(tupled_intervals)

    from intervaltree import IntervalTree
    tree = IntervalTree.from_tuples(tupled_intervals)
    
    md = int(np.round(100.0/(minDiff)))
    print(md)
    for t in range(0,md+1):
        #print(t*maxTime/md*1.0)
        ints = (tree[t*maxTime/md*1.0])
        for i1 in ints:
            start1,end1,x1 = i1
            v1,n1 = x1
            int1 = (v1,start1,end1,tuple(n1))
            for i2 in ints:
                start2,end2,x2 = i2
                v2,n2 = x2
                int2 = (v2,start2,end2,tuple(n2))
                #print(i1)
                graph.add_edge(int1,int2)
    print("first graph constructed")            
    G = nx.Graph()
    print("pruning graph...")
    for e in graph.edges():
        v1,v2 = e
        #G.add_node(v1)
        #G.add_node(v2)
        G.add_edge(v1,v1)
        G.add_edge(v2,v2)
        if kernInt(v1,v2)>eps:
            G.add_edge(v1,v2) 
    print("...done")                   
    return G,ddintm21note




def interval_graph(intervals):
    """Generates an interval graph for a list of intervals given.

    In graph theory, an interval graph is an undirected graph formed from a set
    of closed intervals on the real line, with a vertex for each interval
    and an edge between vertices whose intervals intersect.
    It is the intersection graph of the intervals.

    More information can be found at:
    https://en.wikipedia.org/wiki/Interval_graph

    Parameters
    ----------
    intervals : a sequence of intervals, say (l, r) where l is the left end,
    and r is the right end of the closed interval.

    Returns
    -------
    G : networkx graph

    Examples
    --------
    >>> intervals = [(-2, 3), [1, 4], (2, 3), (4, 6)]
    >>> G = nx.interval_graph(intervals)
    >>> sorted(G.edges)
    [((-2, 3), (1, 4)), ((-2, 3), (2, 3)), ((1, 4), (2, 3)), ((1, 4), (4, 6))]

    Raises
    ------
    :exc:`TypeError`
        if `intervals` contains None or an element which is not
        collections.abc.Sequence or not a length of 2.
    :exc:`ValueError`
        if `intervals` contains an interval such that min1 > max1
        where min1,max1 = interval
    """
    graph = nx.Graph()

    tupled_intervals = [tuple(interval) for interval in intervals]
    graph.add_nodes_from(tupled_intervals)

    while tupled_intervals:
        voice1,min1, max1,note1 = interval1 = tupled_intervals.pop()
        graph.add_edge(interval1, interval1)
        for interval2 in tupled_intervals:
            voice1,min2, max2,note2 = interval2
            if kernInt(interval1,interval2)>=1.0/8.0:
                graph.add_edge(interval1, interval2)
    return graph


def parse_file(xml):
    xml_data = m21.converter.parse(xml)
    score = []
    instruments = []
    for part in xml_data.parts:
        parts = []
        instruments.append(part.getInstrument())
        for note in part.recurse().notesAndRests:
            if note.isRest:
                start = note.offset
                duration = float(note.quarterLength)/4.0
                vol = 32 #note.volume.velocity
                pitch = 60
                parts.append((note,[(pitch,findNearestDuration(duration,durationslist),vol,True)]))
            elif note.isChord:
                chord = []
                vols = ([n.volume.velocity for n in note if not n.volume.velocity is None])
                if len(vols)==0:
                    vol = 64
                else:
                    vol = min(vols)
                for n in note:
                    start = n.offset
                    duration = float(note.quarterLength)/4.0
                    pitch = n.pitch.midi
                    #print(pitch,duration,note.volume)
                    chord.append((pitch,findNearestDuration(duration,durationslist),vol,False))
                parts.append((note,chord))   
                if vol is None:
                    vol = int(64)
                parts.append((note,[(pitch,findNearestDuration(duration,durationslist),vol,False)]))    
            else:
                #print(note)
                start = note.offset
                duration = float(note.quarterLength)/4.0
                pitch = note.pitch.midi
                #print(pitch,duration,note.volume)
                vol = note.volume.velocity
                if vol is None:
                    vol = int(note.volume.realized * 127)
                parts.append((note,[(pitch,findNearestDuration(duration,durationslist),vol,False)]))    
        score.append(parts)        
    return score,instruments


def writeM21Lists(fn,inds,tempo,instruments,title,author,fileType):
    from music21 import chord
    from music21 import stream
    from music21 import duration
    from music21 import clef, metadata
    import music21 as m
    import copy

    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = title
    score.metadata.composer = author
    tm = m.tempo.MetronomeMark(number=tempo)
    score.append(tm)
    lh = m.stream.Part()
    lh.append(m.instrument.Piano())
    rh = m.stream.Part()
    rh.append(m.instrument.Piano()) #Violin())
    
    def extendPart(part,ll):
        for l in ll:
            part.append(l)
        return part
    
    ourparts = []    
    for i in range(len(inds)):
        mypart = m.stream.Part()
        mypart.append(instruments[i])
        mypart = extendPart(mypart, inds[i])
        ourparts.append(mypart)
    
    for part in ourparts:
        score.append(part)
    if fileType == "musicxml":    
        score.write("musicxml",fp=fn) 
    elif fileType == "mid":    
        score.write("mid",fp=fn) 
    
def getCoordinatesOf(intList,kernel=kernInt,nDim=None):
    M0 = np.array([[kernel(t1,t2) for t1 in intList] for t2 in intList])
    #print(M0)
    from sklearn.decomposition import PCA
    from sklearn.decomposition import KernelPCA
    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    scaler = StandardScaler() #MinMaxScaler((0,1))
    KPCA = KernelPCA(n_components=nDim,kernel='precomputed',eigen_solver='randomized')
    
    Ch0 = KPCA.fit_transform((M0))
    #print(Ch0)
    X0 = [x for x in 1.0*Ch0]    
    
    #print(X0)
        
    #X0 = scaler.fit_transform(X0)
    
    #invPitchDict = dict(zip(intList,range(len(intList))))
    return Ch0#, invPitchDict

def run_length_row(row_of_01):
    from itertools import groupby
    ar = row_of_01
    return [(k, sum(1 for i in g)) for k,g in groupby(ar)]

def get_knn_model(X):
    #notes = np.array([[x*1.0 for x in n] for n in notes])
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)
    nbrs = NearestNeighbors( algorithm='ball_tree').fit(X)
    return nbrs

def findBestMatches(nbrs,new_row,n_neighbors=1):
    distances,indices = nbrs.kneighbors([np.array(new_row)],n_neighbors=n_neighbors)
    dx = sorted(list(zip(distances[0],indices[0])))
    #print(dx)
    indi = [d[1] for d in dx]
    #print(indi)
    #print(distances)
    #distances,indices = nbrs.query([np.array(new_row)],k=n_neighbors)
    return indi
    


def processFile(file,funcType="cos",tempo=80,eps=1.0/100.0,title="Music21 Fragment",author="music21",fileType="musicxml"):
    import networkx as nx
    scores,instruments = parse_file(file)
    G, ddintm21note = interval_graph_0(scores,eps=eps)
    import scipy, numpy as np

    numberOfVoices = len(scores)
    ccG = list(nx.connected_components(G))
    print("number of connected components = ", len(ccG))
    all_components_sorted = sorted([[ c for c in ccs] for ccs in ccG])
    rr = sorted(range(len(all_components_sorted)),reverse=False)
    #shuffle(all_components_sorted)
    inds = []
    import copy
    for v in range(numberOfVoices):
        inds.append([])
    div = 2*len(rr) 
    #funcType="cos"
#funcType="blancmange"
#funcType="cellerier"
#funcType="weierstrass"
    if funcType=="cos":
        ll = [(np.cos((j/(1/2*div)+0.5)*2*np.pi)/2.0+0.5)*np.exp(j/div) for j in range(0,div+1)]
    elif funcType == "blancmange":
        import numpy as np
        def S(x):
            return abs(x-round(x))
        def B(x):
            return sum([S(2**i*x)/2**i for i in range(100)])

        ll = [B(j/div) for j in range(0,div+1)]    
    elif funcType=="cellerier":
        div = 2*len(rr)
        a = 2
        def cellerier(x,N,a=2):
            y = np.zeros((1,div))
            for n in range(1,N):
                y = y + np.sin(a**n*np.pi*x)/a**n
            return y[0].tolist()
        x = np.linspace(-1,1,div)
        ll = cellerier(x,50,a=a)  
        print(ll)
    elif funcType=="weierstrass":
        def weierstrass(x, N):
            y = np.zeros((1,div))
            for n in range(1,N):
                y = y + np.cos(3**n*np.pi*x)/2**n
            return y[0].tolist() 
        x = np.linspace(-1,1,div)
        ll = weierstrass(x,500)       
 
    print("computing coordinates...")   
    XNotes = (getCoordinatesOf(all_components_sorted,kernel=kernConnectedComponent(kernIntChord),nDim=None))        
    print("...done")
#nbrs = get_knn_model(Ch0)
#print(nbrs)


#XNotes  = getCoordinatesOf(ints,kernel=kernInt,nDim=None)
  
    import scipy, numpy as np

    print("computing knn-model...")
    nbrsNotes = get_knn_model(XNotes)
    print("...done")
        #dicts = [xnotes]
    
    new_notes = []
#div = 30*len(ints)
    import numpy as np , bezier
    XXBezier = XNotes
    nodes = XXBezier.T
    print("computing bezier curve...")
    curve = bezier.Curve(nodes, degree=XXBezier.shape[0]-1)
    print("...done")
    nComp = len(all_components_sorted)

#ss = [(j/div*1.0)**2 for j in range(0,div+1)]
    ss = [(l-min(ll))/(max(ll)-min(ll)) for l in ll]
    print(curve)
    print("evaluating at bezier curve..")            #print(ss)
    for s in ss:
        new_row = curve.evaluate(s)
        indi = findBestMatches(nbrsNotes,new_row.T[0])
        new_notes.append(indi[0])
    print("...done")        
    #for t in run_length_row(new_notes):
    for t in new_notes:
        #ccs = all_components_sorted[t[0]]
        ccs = all_components_sorted[t]
        for v in range(numberOfVoices):
        #print(v,[(ddintm21note[c],c) for c in sorted([c for c in ccs if c[0]==v])])]
            cv =  sorted([c for c in ccs if c[0]==v])
            import random
            if random.randint(1,100)/100.0<=0.25:
                cv.reverse()
            ccv = [copy.deepcopy(ddintm21note[c]) for c in cv]
            c3v = []
            for c in ccv:
                if not c.isRest and random.randint(1,100)/100.0<=0.25:
                    c = c.transpose("M3")
                if not c.isRest and random.randint(1,100)/100.0<=0.25:
                    c = c.transpose("P4")
                c3v.append(c)    
            inds[v].extend(c3v)
            print(t,v)
    #print(inds)
    #ffn = file+"-"+funcType+"-"+str(eps)+".xml"
    ffn = file+"-"+funcType+"-"+str(eps)+".mid"
    writeM21Lists(fn=ffn,inds=inds,tempo=tempo,instruments=instruments,title=title,author=author,fileType=fileType)    
    return ffn

#file = "./tiersen.mid"
#processFile(file,funcType="cos",tempo=110,eps=1.0/10.0**4)        
