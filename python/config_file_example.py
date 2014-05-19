import cPickle
import yaml
import wormtracker as wt
import wormtracker.config as wtc

pickleFile = 'D:\\N2_a_b_day_7.dat'
configFile = 'D:\\test.yml'

# load pickled WormVideo
with open(pickleFile, 'rb') as f:
    wv = cPickle.load(f)

# save WormVideo to YAML configuration file
# the .YAML file is a human-readable configuration format
# yaml.load(f) returns a nested set of lists/dictionaries
with open(configFile, 'w') as f:
    wtc.saveWormVideo(wv, f)

# load WormVideo to YAML configuration file
with open(configFile, 'r') as f:
    wvs = wtc.loadWormVideos(f)

# wvs[0] should be equivalent to the original wv object

"""
Example output: test.yml

systemSettings: {hdf5Path: 'C:\hdf5\', libavPath: 'C:\libav\bin\'}
videos:
- regions:
  - cropRegion: !!python/tuple [461, 157, 311, 393]
    foodCircle: !!python/tuple [115.94665271966528, 156.94665271966528, 97.49804815215789]
    strainName: N2
    wormName: '1'
  - cropRegion: !!python/tuple [491, 635, 293, 441]
    foodCircle: !!python/tuple [88.33786610878656, 218.84675732217568, 101.49190795222891]
    strainName: N2
    wormName: '2'
  - cropRegion: !!python/tuple [413, 1194, 363, 447]
    foodCircle: !!python/tuple [178.66213389121336, 214.81746861924682, 96.68284794292089]
    strainName: N2
    wormName: '3'
  - cropRegion: !!python/tuple [395, 1725, 384, 396]
    foodCircle: !!python/tuple [166.6464435146443, 208.89121338912133, 97.01203384388343]
    strainName: N2
    wormName: '4'
  - cropRegion: !!python/tuple [896, 157, 396, 396]
    foodCircle: !!python/tuple [200.6066945606694, 197.49999999999997, 103.8512019645039]
    strainName: N2
    wormName: '5'
  - cropRegion: !!python/tuple [890, 638, 387, 447]
    foodCircle: !!python/tuple [198.84466527196656, 235.8582635983263, 97.028485024507]
    strainName: N2
    wormName: '6'
  - cropRegion: !!python/tuple [884, 1197, 387, 453]
    foodCircle: !!python/tuple [185.8922594142259, 189.2766736401673, 91.45415645110153]
    strainName: N2
    wormName: '7'
  - cropRegion: !!python/tuple [875, 1731, 378, 411]
    foodCircle: !!python/tuple [203.54707112970706, 207.14958158995813, 96.1862703354842]
    strainName: N2
    wormName: '8'
  - cropRegion: !!python/tuple [1513, 154, 381, 420]
    foodCircle: !!python/tuple [170.2301255230126, 177.64853556485355, 94.51352003796741]
    strainName: N2
    wormName: '9'
  - cropRegion: !!python/tuple [1509, 656, 393, 459]
    foodCircle: !!python/tuple [150.3880753138075, 219.39748953974896, 96.03260627078504]
    strainName: N2
    wormName: '10'
  - cropRegion: !!python/tuple [1500, 1215, 387, 456]
    foodCircle: !!python/tuple [178.69037656903768, 207.2280334728033, 98.02208104862247]
    strainName: N2
    wormName: '11'
  - cropRegion: !!python/tuple [1500, 1756, 381, 372]
    foodCircle: !!python/tuple [192.91841004184099, 202.0376569037657, 89.50319476835287]
    strainName: N2
    wormName: '12'
  - cropRegion: !!python/tuple [2014, 185, 359, 399]
    foodCircle: !!python/tuple [172.73953974895403, 170.82792887029285, 96.63238912865307]
    strainName: N2
    wormName: '13'
  - cropRegion: !!python/tuple [2008, 668, 369, 441]
    foodCircle: !!python/tuple [158.62866108786613, 230.3791841004184, 91.91279027950345]
    strainName: N2
    wormName: '14'
  - cropRegion: !!python/tuple [1999, 1227, 366, 450]
    foodCircle: !!python/tuple [160.1412133891214, 251.56589958158995, 89.4428880705747]
    strainName: N2
    wormName: '15'
  - cropRegion: !!python/tuple [1996, 1762, 363, 359]
    foodCircle: !!python/tuple [167.85669456066944, 203.40899581589957, 100.7152789673288]
    strainName: N2
    wormName: '16'
  videoSettings:
    backgroundDiskRadius: 5
    expectedWormLength: 1000
    expectedWormWidth: 50
    frameRate: 11.5
    pixelsPerMicron: 0.05
    storeFile: 2014-04-14_n2_a_b_day_7_processed.h5
    threshold: 0.9
    videoFile: //system-biologysrv.amolf.nl/users/Koers/worm_videos/day_7/2014-04-14_n2_a_b_day_7.avi
    wormAreaThresholdRange: [0.5, 1.5]
    wormDiskRadius: 2
"""
