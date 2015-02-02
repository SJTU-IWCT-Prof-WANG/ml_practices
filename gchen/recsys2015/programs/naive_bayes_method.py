#coding=utf8
#
# Filename:    naive_bayes_method.py
# Author:      Ge CHEN <princhene1991@126.com>
# Date:        2015-02-01
#
# Use naive bayesian method to predict user actions
#
# @classes
# --------
#     NaiveBayesModelCreation: generate model parameters according
#                            to bayesian features
#     NaiveBayesPrediction: predict user actions with test data
#     ResultsEvaluation:  calculate score and precision, with the
#                         score criterion on: 
#                         http://2015.recsyschallenge.com/challenge.html

import cPickle
import math
import numpy
import os
import sys

class NaiveBayesModelCreation:

    def __init__( self, clicks_file, buys_file ):
        """Initiate clicks and buys dict file
        """
        self.clicks_dict_file = clicks_file
        self.buys_dict_file = buys_file
        self.parameters = {}


    def match_clicks_buys( self, clicks, buys ):
        """Given a list of clicks and buys in the
        same session, return the corresponding seq:
            clicks_buys = { x_i: {'count':n_i, 'buy': b_i} }
        where x_i ~ item_id and b_i ~ [0,1] meaning
        not buy/ buy.
        """
        clicks_buys = {}; items_bought_set = set()

        for buy_record in buys:
            items_bought_set.add( buy_record[ 'item' ] )

        for click in clicks :
            item = click[ 'item' ]
            if item in clicks_buys:
                clicks_buys[ item ][ 'count' ] += 1
            else:
                if item in items_bought_set:
                    clicks_buys[ item ] = { 'count': 1, 'buy': 1 }
                else:
                    clicks_buys[ item ] = { 'count': 1, 'buy': 0 }

        return clicks_buys

    def generate_params( self, clicks_buys ):
        """ Generate parameters for bayes inference:
            P( Xi, buy= 0/1 ) = [ X_j , X_j+1, X_j+2 ... ]
        """
        items = clicks_buys.keys()
        for item in items:
            buy = clicks_buys[ item ][ 'buy' ]
            self.parameters.setdefault( item, {} )
            self.parameters[ item ].setdefault( buy, {} )
            self.parameters[ item ].setdefault( 1-buy , {} )
            for sub_item in items:
                self.parameters[ item ][ buy ].setdefault( sub_item, 1 )
                self.parameters[ item ][ 1-buy ].setdefault( sub_item, 1 )
                self.parameters[ item ][ buy ][ sub_item ] += clicks_buys[ sub_item ][ 'count' ]

    def unify_params( self ):
        """ Refresh parameters to avoid pick point divided by 0 
        """
        print "\n\t Start unifying parameters to probabilities"
        progress = 0
        for item in self.parameters:
            progress += 1
            if progress % 1000 == 0:
                sys.stdout.write( "\r\t\t progress:" + str( progress ) )
                sys.stdout.flush()

            for buy in [0, 1]:
                cur_sum = sum( self.parameters[ item ][ buy ].values() )
                for sub_item in self.parameters[ item ][ buy ]:
                    cur_value = self.parameters[ item ][ buy ][ sub_item ] + 0.0
                    self.parameters[ item ][ buy ][ sub_item ] = cur_value / cur_sum
        print "\n\t Unifying parameters Finished~~"

    def create( self ):
        """ Load data from dictionary, create model
        on [Cn] and [Cn-1,Cn] and store it to a directory
        """
        # Load clicks dictionary
        print "\n\t Start loading clicks dict file"
        with open( self.clicks_dict_file, 'rb' ) as fstream:
            clicks_dict = cPickle.load( fstream )
        print "\n\t Load clicks Finished~~"

        # Load buys dictionary
        print "\n\t Start loading buys dict file"
        with open( self.buys_dict_file, 'rb' ) as fstream:
            buys_dict = cPickle.load( fstream )
        print "\n\t Load buys Finished~~"

        # for each session get clicks sequence and corresponding buys sequence
        print "\n\t Start generating parameters"
        progress = 0
        for session in clicks_dict:
            progress += 1
            if progress % 10000 == 0:
                sys.stdout.write( "\r\t\t progress:" + str( progress ) )
                sys.stdout.flush()

            clicks = clicks_dict[ session ]
            buys = buys_dict.get( session, [] )

            clicks_buys = self.match_clicks_buys( clicks, buys )

            self.generate_params( clicks_buys )

        self.unify_params()
        print "\n\t Parameters generation finished~"


    def store_params( self, dir_2_store ):
        """ Store parameters dict to the directory for later use
        """
        print "\n\t Start storing parameters"
        with open( dir_2_store + os.sep + 'naive_bayes_params.dict' , 'wb' ) as fstream:
            cPickle.dump( self.parameters, fstream, -1 )
        print "\n\t Parameters storage finished~"


class NaiveBayesPrediction:

    def __init__( self, params_file, test_file ):
        """ Initiate class with params/test file path
        and the initial probability distribution on
        single and neighbor estimator.
        """
        self.params_file = params_file
        self.test_file = test_file
        self.results = {}

    def load_params( self ):
        print "\t Start Load parameters from sequence model"
        with open( self.params_file, 'rb' ) as fstream:
            self.parameters = cPickle.load( fstream )
        print "\t Load parameters Finished ~~"

    def predict( self, clicks_dict ):
        """ Given a series of clicks in one session
        generate the most probable buys dict
            B = [item_1, item_2, .., item_j]
        j items are assumed to be bought
        """
        buys = []
        for item in clicks_dict:
            prob_buy, prob_not_buy = 0, 0
            item_params = self.parameters.get( item, {} )

            not_buy_params = item_params.get( 0, {} )
            buy_params = item_params.get( 1, {} )
            for sub_item in clicks_dict:
                if sub_item in not_buy_params:
                    prob_not_buy += math.log( not_buy_params[ sub_item ] ) \
                                     * clicks_dict[ sub_item ]
                    prob_buy += math.log( buy_params[ sub_item ] ) \
                                     * clicks_dict[ sub_item ]

            if prob_not_buy < prob_buy:
                buys.append( item )

        return buys

    def do_task( self, dir_2_store ):
        print "\t Start Load test sessions dict from sequence model"
        with open( self.test_file, 'rb' ) as fstream:
            test_dict = cPickle.load( fstream )
        print "\t Load test sessions dict Finished ~~"
        print "\t Number of sessions in test dict:", len( test_dict.keys() )

        progress = 0
        for session in test_dict:
            progress += 1
            if progress % 10000 == 0:
                sys.stdout.write( "\r\t\t progress:" + str( progress ) )
                sys.stdout.flush()

            clicks = test_dict[ session ]
            clicks_dict = {}
            buys = self.predict( clicks_dict )
            if buys:
                self.results[ session ] = buys

        #Store prediction results to a csv
        print "\t Start storing results to ", dir_2_store
        with open( dir_2_store + os.sep + 'naive_bayes_results' , 'w' ) as fstream:
            for session in self.results:
                fstream.write( session + ";" + \
                               ",".join( self.results[ session ] ) \
                               + '\n' )
        print "\t Storage of results Finished ~~"


class ResultsEvaluation:

    def __init__( self, test_file, results_file, answers_file ):
        self.score = 0.0
        self.test_file = test_file
        self.results_dict = {}
        self.answers_dict = {}
        print "\n\t Start Load results"
        with open( results_file, 'r' ) as fstream:
            for line in fstream:
                [ session, items ] = line.split( ';' )
                items = items.split( ',' )
                self.results_dict[ session ] = items
        print "\t Load results Finished ~~"
        print "\t Start Load answers"
        with open( answers_file, 'rb' ) as fstream:
            answers_orig = cPickle.load( fstream )
            for session in answers_orig:
                self.answers_dict[ session ] = []
                for record in answers_orig[ session ]:
                    self.answers_dict[ session ].append( record[ 'item' ] )
        print "\t Load answers Finished ~~"


    def cal_score( self ):
        """ calculate score of the current results
        """
        S = 9249729
        Sb = len( self.answers_dict.keys() ) + 0.0
        increment_unit = Sb / S
        progress = 0
        for session in self.results_dict:
            progress += 1
            if progress % 10000 == 0:
                sys.stdout.write( "\r\t\t progress:" + str( progress ) )
                sys.stdout.flush()


            if session in self.answers_dict:
                self.score += increment_unit
                results = set( self.results_dict[ session ] )
                answers = set( self.answers_dict[ session ] )
                self.score += \
                        (len( results.intersection( answers ) ) + 0.0) \
                        / len( results.union( answers ) )
            else:
                self.score -= increment_unit

        print "\n\t The score range: [", Sb*Sb/S - Sb, ",", Sb*Sb/S + Sb, "]"
        print "\n\t The final score is : ", self.score

    def cal_precision( self ):
        """ calculate precision of the current results
        """
        S = 9249729.0
        Sb = len( self.answers_dict.keys() ) + 0.0
        results_set = set( self.results_dict.keys() )
        answers_set = set( self.answers_dict.keys() )

        union_amount = len( results_set.union( answers_set ) )
        inter_amount = len( results_set.intersection( answers_set ) )
        precision = ( inter_amount + S - union_amount ) / S

        print "\n\t The score range: [", 0, ",", 1, "]"
        print "\n\t The final precision is : ", precision


def main():
    clicks_file = "../median_datasets/training_sessions.dict"
    buys_file = "../median_datasets/buys_sessions.dict"
    tests_file = "../median_datasets/tests_sessions.dict"
    """
    # generate parameters and store them
    sequence_model = NaiveBayesModelCreation( clicks_file, buys_file )
    sequence_model.create()
    dir_2_store = "../median_datasets"
    sequence_model.store_params( dir_2_store )
    """
    """
    # use model to predict results
    prediction = NaiveBayesPrediction( "../median_datasets/naive_bayes_params.dict", clicks_file )
    prediction.load_params()
    dir_2_store = "../results_datasets"
    prediction.do_task( dir_2_store )
    """
    # calculate score
    result_file = "../results_datasets/naive_bayes_results"
    evaluation = ResultsEvaluation( clicks_file, result_file, buys_file)
    #evaluation.cal_score()
    evaluation.cal_precision()


if __name__ == "__main__":
    main()
