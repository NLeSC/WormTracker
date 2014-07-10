import h5py
import yaml

dataset = 'repacked.h5'
prefix = 'sdsbox'

with h5py.File(dataset, 'r') as f:
    strains = f.keys()
    for strain in strains:
        videos = f[strain]['videos'].keys()
        for video in videos:
            configName = '{0}_{1}_{2}.yml'.format(prefix, strain, video)
            with open(configName, 'w') as cf:
                vd = f[strain]['videos'][video]
                expWormLength = float(vd['expected_worm_length'][0])
                expWormWidth = float(vd['expected_worm_width'][0])
                pixelsPerMicron = float(vd['pixels_per_um'][0])
                threshold = float(vd['threshold'][0])
                videoFile = str(vd.attrs['FileName'])
                wormDiskRadius = round(expWormWidth*pixelsPerMicron/2.)
                config = {
                    'systemSettings': {
                        'hdf5path': 'C:\\hdf5'
                    },
                    'processingSettings': {
                        'smoothing': 0.05
                    },
                    'postprocessingSettings': {
                        'filterByWidth': True,
                        'filterByLength': True,
                        'widthThreshold': (0.5, 1.5),
                        'lengthThreshold': (0.8, 1.2),
                        'max_n_missing': 10,
                        'max_d_um': 10.,
                        'allowedSegmentSize': (150, 500),
                        'headMinSpeed': 40.,
                        'headMinLeading': 2,
                        'headMinRelSpeed': 1.2
                    },
                    'videos': [{
                        'regions': [],
                        'videoSettings': {
                            'backgroundDiskRadius': 5,
                            'expectedWormLength': expWormLength,
                            'expectedWormWidth': expWormWidth,
                            'frameRate': 11.5,
                            'pixelsPerMicron': pixelsPerMicron,
                            'numberOfPosturePoints': 50,
                            'threshold': threshold,
                            'storeFile': '{0}_{1}_{2}.h5'.format(prefix,
                                                                 strain,
                                                                 video),
                            'videoFile': videoFile,
                            'wormAreaThresholdRange': (0.5, 1.5),
                            'wormDiskRadius': wormDiskRadius
                        }
                    }]
                }
                td = f[strain]['trajectories']
                worms = [w for w in td.keys()
                         if int(td[w]['video']['video_id'][0]) == int(video)]
                for w in worms:
                    crop = tuple(int(i) for i in td[w]['video']['crop_region'][...].astype('i8'))
                    config['videos'][0]['regions'].append({
                            'cropRegion': crop,
                            'strainName': str(strain),
                            'wormName': str(w)
                        })
                yaml.dump(config, cf)
