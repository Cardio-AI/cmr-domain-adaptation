import logging
from medpy.metric.binary import hd, dc,jc,precision,recall
from src.visualization.Visualize import show_slice
import random
import numpy as np
from numpy import newaxis


def calc_score_per_label(pred, gt, score_f):

    # define label and labelname mapping
    labels = [None, 0, 1, 2, 3]
    l_names = ['total', 'background','RV', 'Myo', 'LV']

    for label, l_name in zip(labels,l_names):

        if label == None: # calc the total score
            logging.info('{} score: {:.3f}'.format(l_name, score_f(pred, gt)))

        else: # check if pred and gt have values for this label, else return 0
            if pred[..., label].max() and gt[..., label].max():
                logging.info('{} score: {:.3f}'.format(l_name, score_f(pred[..., label], gt[..., label])))
            else:
                logging.info('{} score: 0.000'.format(l_name))

                
def sanity_check(generator, model, f_size=(15,5), scores_f = [dc, hd, jc], s_names = ['dice', 'hausdorff distance', 'jaccard index']):
    
    # take one random (image, gt) example from a generator
    # predict with the given model
    # calculate scores
    
    assert(len(scores_f) == len(s_names))
    ix = random.randint(0, len(generator))
    ib = random.randint(0, generator.batch_size-1)

    # get random image
    img = generator.__getitem__(ix)[0][ib]
    pred = model.predict(img[newaxis, ...])
    gt = generator.__getitem__(ix)[1][ib]
    
    # threshold prediction
    pred = pred[0,...]
    pred = (pred>0.5).astype(np.bool)
    gt = gt.astype(np.bool)
    
    logging.info('Ground-Truth')
    show_slice(img=img, mask=gt, f_size=f_size)
    logging.info('Prediction')
    show_slice(img=img, mask=pred, f_size=f_size)

    try:
        logging.info('shape image: {}'.format(img.shape))
        logging.info('type image: {}'.format(img.dtype))
        logging.info('shape pred mask: {}'.format(pred.shape))
        logging.info('type pred mask: {}'.format(pred.dtype))
        logging.info('shape gt mask: {}'.format(gt.shape))
        logging.info('type gt mask: {}'.format(gt.dtype))
        
        for score_f, s_name in zip(scores_f, s_names):
        
            logging.info('{}:'.format(s_name))
            calc_score_per_label(pred,gt,score_f)
            
        return pred, gt
    
    except Exception as e:
        logging.info(str(e))
        return pred, gt
    
