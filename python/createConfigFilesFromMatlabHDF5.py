import h5py
import yaml

dataset = 'repacked.h5'
prefix = 'sdsbox'

with h5py.File(dataset, 'r') as f:
    strains = f.keys()
    for strain in strains:
        videos = dataset[strain]['videos'].keys()
        for video in videos:
            configName = '{0}_{1}_{2}.yml'.format(prefix, strain, video)
            with open(configName, 'w') as cf:
                vd = dataset[strain]['videos'][video]
                expWormLength = vd['expected_worm_length'][...]
                expWormWidth = vd['expected_worm_width'][...]
                pixelsPerMicron = vd['pixels_per_um'][...]
                threshold = vd['threshold'][...]
                videoFile = vd.attr['FileName']
                config = {
                    'systemSettings': {
                        'hdf5path': 'C:\hdf5'
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
                            'wormAreaThresholdRange': [0.5, 1.5],
                            'wormDiskRadius': 2
                        }
                    }]
                }
                td = dataset[strain]['trajectories']
                worms = [w for w in td.keys()
                         if int(td[w]['video']['video_id'][...]) == int(video)]
                for w in worms:
                    crop = tuple(td[w]['video']['crop_region'][...].astype('i8'))
                    config['videos'][0]['regions'].append({
                            'cropRegion': crop,
                            'strainName': strain,
                            'wormName': w
                        })
                yaml.dump(config, cf)
