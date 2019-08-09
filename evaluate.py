# --------------------------------------------------------
# Dense-Captioning Events in Videos Eval
# Copyright (c) 2017 Ranjay Krishna
# Licensed under The MIT License [see LICENSE for details]
# Written by Ranjay Krishna
# --------------------------------------------------------

import argparse
import string
import json
import sys
sys.path.insert(0, './coco-caption') # Hack to allow the import of pycocoeval

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from sets import Set
import numpy as np
import csv
import re

def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

class ANETcaptions(object):
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                verbose=False):
        # Check that the gt and submission files exist and load them
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')

        self.verbose = verbose
        self.ground_truth = self.import_ground_truth(ground_truth_filename)
        self.prediction = self.import_prediction(prediction_filename)
        self.tokenizer = PTBTokenizer()

        # Set up scorers, if not verbose, we only use the one we're
        # testing on: METEOR
        if self.verbose:
            self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(),"METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr"),
                (Spice(), "SPICE")
            ]
        else:
            self.scorers = [(Meteor(), "METEOR")]

    def import_prediction(self, prediction_filename):
        if self.verbose:
            print "| Loading submission..."
        results = json.load(open(prediction_filename))
        print('len of results:', len(results))
        return results

    def import_ground_truth(self, filename):
        gt = {}
        self.n_ref_vids = Set()
        id = 1
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file,delimiter='\t')
            for row in csv_reader:
                # remove punctuation but keep possessive because we want to separate out characfter names
                caption = row[5]
                gt[id] = caption
                id+=1
        self.n_ref_vids.update(gt.keys())
        if self.verbose:
            print "| Loading GT. file: %s, #videos: %d" % (filename, len(self.n_ref_vids))
        return gt

    def check_gt_exists(self, vid_id):
        return vid_id in self.ground_truth

    def get_gt_vid_ids(self):
        return self.ground_truth.keys()

    def evaluate(self):
        # This method averages the tIoU precision from METEOR, Bleu, etc. across videos
        res = {}
        gts = {}
        gt_vid_ids = self.get_gt_vid_ids()

        unique_index = 0

        # video id to unique caption ids mapping
        vid2capid = {}

        cur_res = {}
        cur_gts = {}

        for info in self.prediction:
            # If the video does not have a prediction, then we give it no matches
            # We set it to empty, and use this as a sanity check later on

            # If we do have a prediction, then we find the scores based on all the
            # valid tIoU overlaps
            vid_id = info['video_id']
            if vid_id in gt_vid_ids:
                pred = info['caption']
                gt_caption = re.sub(r'[.!,;?]', ' ', self.ground_truth[vid_id].lower())
                cur_res[unique_index] = [{'caption': remove_nonascii(pred)}]
                cur_gts[unique_index] = [{'caption': remove_nonascii(gt_caption)}] # for now we use gt proposal
                vid2capid[vid_id] = unique_index
                unique_index += 1

        # Each scorer will compute across all videos and take average score
        output = {}

        # call tokenizer here for all predictions and gts
        res = self.tokenizer.tokenize(cur_res)
        gts = self.tokenizer.tokenize(cur_gts)

        for scorer, method in self.scorers:
            print 'computing %s score...' % (scorer.method())
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    output[m] = sc
            else:
                output[method] = score
        return output

def main(args):
    # Call coco eval
    evaluator = ANETcaptions(ground_truth_filename=args.reference,
                             prediction_filename=args.submission,
                             verbose=args.verbose)
    output = evaluator.evaluate()
    for metric, score in output.items():
        print '| %s: %2.4f'%(metric, score)
    # output[metric] = 100 * sum(score) / float(len(score))
    json.dump(output,open(args.output,'w'))
    print(output)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate the results stored in a submissions file.')
    parser.add_argument('-s', '--submission', type=str,  default='sample_submission.json',
                        help='sample submission file for LSMDC.')
    parser.add_argument('-r', '--reference', type=str,
                        help='reference csv with ground truth captions')
    parser.add_argument('-o', '--output', type=str,  default='result.json',
                        help='output file with final language metrics.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print intermediate steps.')
    args = parser.parse_args()

    main(args)
