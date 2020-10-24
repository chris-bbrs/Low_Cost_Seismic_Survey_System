from warnings import filterwarnings; filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path, makedirs, _exit
import shutil
from sklearn.linear_model import LinearRegression

# /....................................................................................................
def findpeaks(y, ratio=0.005):
    s = y
    st = [s.argmax()]
    for j in range(2):
        peak = y.argmax()
        i = peak-1
        while i > 3 and np.any(abs((np.diff(y[i-3:i+1]) / np.diff([y[peak],y[i]])) * 100) > ratio):
            if np.diff([y[i-1],y[i]]) < 0:
                if abs((np.diff([y[i-1],y[i]]) / np.diff([y[peak],y[i]])) * 100) >= 1: #60
                    st.append(s.argmax()+peak-i) if j else st.append(s.argmax()-peak+i)
                    y = -y
                elif np.diff([y[i-2],y[i-1]]) < 0:
                    if abs((np.diff([y[i-2],y[i]]) / np.diff([y[peak],y[i]])) * 100) >= 2: #50
                        st.append(s.argmax()+peak-i) if j else st.append(s.argmax()-peak+i)
                        y = -y
                    elif np.diff([y[i-3],y[i-2]]) < 0:
                        st.append(s.argmax()+peak-i) if j else st.append(s.argmax()-peak+i)
                        y = -y
            i = i - 1
        y = np.flipud(s)
    st.sort()
    return st
# /...................................................

def fbreak(s, peak):
    y = s[:peak+1]
    peak = len(y)-1
    i = peak-1

    if np.diff(y[i-1:i+1]) < 0: y = -y
    it = 5
    while i > it and ((np.diff(y[i-it:i+1]).sum()/it) / (np.diff(y[i-it:peak+1]).sum()/(peak-i+it)) * 100) > 35 and np.diff(y[i-1:i+1]) >= 0:
        i -= 1
    else:
        if i == 5: it -= 1
        gg = np.diff(y[i-(it+1):i+1])
        maxind = gg.argmax()
        for j in range(1,len(gg)):
            if gg[j] <= gg[maxind] and gg[j] >= 0:
                maxind = j
        i-= len(gg)-1 - maxind
    
    return i
# /.................................................

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
# ..................................................

#dir is not keyword
def makemydir(folder):
    try:
        makedirs(folder)
    except OSError:
        pass
# ..................................................

def cut_points(n, already_cut=None):
    # The first cut point is at 0 
    if already_cut is None:
        already_cut = [0]

    # We can cut at all places between the last cut plus 3 
    # and the length minus 3, and yield recursively the solutions for each choice
    for i in range(already_cut[-1]+3, n-2): # already_cut[-1]+2, n-1
        cuts = already_cut[:] + [i]
        yield from cut_points(n, cuts)

    # When we tried all cut points and reached the total length, we yield the cut points list 
    yield already_cut[:] + [n]
# ......................................................

def all_possible_sublists(data):
    n = len(data)
    for cut in cut_points(n):
        yield [data[cut[i]:cut[i+1]] for i in range(len(cut)-1)]
# /.......................................................................................................


print('Press "e" to exit')
while 1:
    directory = input('\n Directory: ')
    
    while not path.exists(directory) or not directory.endswith('.asc'):
        if directory in 'EeΕε': _exit(0)
        print(" File doesn't exist")
        directory = input('\n Directory: ')

    reader = pd.read_csv(directory, header=None, skiprows=34, nrows=1, sep=None, iterator=True, chunksize=1, engine='python')
    delim = reader._engine.data.dialect.delimiter
        
    shma = pd.read_csv(directory, header=None, skiprows=33, sep=delim).dropna(axis='columns').T.values

    user_input = input(' Hertz: ')
    try:
        hz = int(user_input)
    except ValueError:
        hz = user_input

    shma = shma[np.where(np.array([np.std(s) for s in shma]) > 1 )]

    ratio = 0.05
    fbs = [fbreak(shma[0], findpeaks(shma[0], ratio)[0])]
    for i in range(1, len(shma)):
        st = findpeaks(shma[i], ratio)
        for p in st:
            fb = fbreak(shma[i], p)
            if fb > fbs[-1]:
                break
        fbs.append(fb)

    shma = [s-s[fbs[0]].mean() for s in shma]

    if isinstance(hz, str):
        x = [range(len(s)) for s in shma]
    else:
        x = [np.arange(len(s)) / hz for s in shma]
        fbs = np.array(fbs) / hz

    if len(shma) <= 10:
        colors = ['C'+ str(i) for i in range(10)]

        [plt.plot(x[i], shma[i], color=colors[i], label=i) for i in range(len(shma))]
        plt.legend()
        [plt.axvline(fbs[i], color=colors[i], ls='--') for i in range(len(fbs))]
        plt.pause(0.001)
        plt.show()

        [print(' ',c+1,'-',f) for c,f in enumerate(fbs)]

        user_input = input(' Distance: ')
        dist = [float(c.strip()) for c in user_input.split(',') if is_number(c)]
        while len(dist) != len(fbs) and user_input not in 'EeΕε':
            print(' Wrong Distances')
            user_input = input(' Distance: ')
            dist = [float(c.strip()) for c in user_input.split(',') if is_number(c)]  

        if user_input in 'EeΕε':
            restart = True
        else:
            restart = False     


        # np.savetxt(path.splitext(directory)[0]+'_Times'+'.csv', np.c_[fbs,dist], delimiter=",", fmt="%s")
    else:
        print(' Too many waves')
        break

    if  not restart:
        np.savetxt(path.splitext(directory)[0]+'_Times'+'.csv', np.c_[fbs,dist], delimiter=",", fmt="%s")

        subseq = [[np.reshape(s, (len(s), 1)) for s in sub] for sub in all_possible_sublists(fbs)]
        subdist = [[np.reshape(s, (len(s), 1)) for s in sub] for sub in all_possible_sublists(dist)]

        modelr, r_adj= [], []
        for i in range(len(subseq)):
            modelr.append([LinearRegression().fit(subdist[i][j], subseq[i][j]) for j in range(len(subseq[i]))])
            r_adj.append([1-(1-modelr[i][j].score(subdist[i][j], subseq[i][j]))*(len(subseq[i][j])-1)/(len(subseq[i][j])-subdist[i][j].shape[1]-1) for j in range(len(subseq[i]))])

        new_fold = 'images'
        if path.exists(new_fold):
            shutil.rmtree(new_fold)
        makemydir(new_fold)
        for x in range(len(subseq)):
            [plt.scatter(subdist[x][j], subseq[x][j], color=colors[j], label=str(round(1/modelr[x][j].coef_[0][0], 3))+' m/s, '+str(round(r_adj[x][j], 3))) for j in range(len(subseq[x]))]
            plt.legend()
            [plt.plot(subdist[x][j], modelr[x][j].predict(subdist[x][j]), color=colors[j]) for j in range(len(modelr[x]))]
            plt.xlabel('Distance (m)')
            plt.ylabel('Time (sec)')
            plt.savefig(path.join('images', str(x+1)+'.png'), bbox_inches='tight')
            plt.close()

        once = False
        user_input = input(' Best Image: ')
        while user_input not in 'EeεΕ':
            if is_number(user_input):
                best_im = int(user_input) - 1
            else:
                best_im = -1

            if best_im >= 0 and best_im < len(subseq):
                once = True
                a, b = [], []
                for x in range(len(subseq)):
                    a.append([modelr[x][j].coef_[0][0] for j in range(len(subseq[x]))])
                    b.append([modelr[x][j].intercept_[0] for j in range(len(subseq[x]))])

                xco = np.round([np.absolute(b[best_im][j]-b[best_im][j+1])/np.absolute(a[best_im][j]-a[best_im][j+1]) for j in range(len(subseq[best_im])-1) if len(modelr[best_im]) > 1], 3)
                xcrit = np.round([(xco[0] * 1/a[best_im][i]) / (1/a[best_im][i+1]+1/a[best_im][i]) for i in range(len(a[best_im])-1)], 3)
                h = np.round([(xco[0]/2) * np.sqrt((1/a[best_im][i+1] - 1/a[best_im][i]) / (1/a[best_im][i+1] + 1/a[best_im][i])) for i in range(len(a[best_im])-1)], 3)
                v_n = best_im
                print(' xco:  ', xco)
                print(' xcrit:', xcrit)
                print(' h:    ', h)
            else:
                print(' Not valid image number')
            user_input = input('\n Best Image: ')

        if once:
            np.savetxt(path.splitext(directory)[0]+'_Vars'+'.csv', np.c_[np.append(np.round(1/np.array(a[v_n]),3),[xco,xcrit,h])].T, header=','.join(np.append(['v'+str(i) for i in range(1,len(subseq[v_n])+1)], ['Xco','Xcrit','h'])), delimiter=",", fmt="%s", comments='')
