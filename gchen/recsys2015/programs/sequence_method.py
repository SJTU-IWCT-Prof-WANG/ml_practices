#coding=utf8
#
# Filename:    sequence_method.py
# Author:      Ge CHEN <princhene1991@126.com>
# Date:        2015-01-31
#
# Use neighboring sequence inference to predict user actions
#
# @classes
# --------
#     SequenceModelCreation: generate model parameters according
#                            to sequence features
#     SequencePrediction: predict user actions with test data
#     ResultsEvaluation:  calculate score and precision, with the
#                         score criterion on: 
#                         http://2015.recsyschallenge.com/challenge.html

import cPickle
import numpy
import os
import sys

class SequenceModelCreation:

    def __init__( self, clicks_file, buys_file ):
        """Initiate clicks and buys dict file
        """
        self.clicks_dict_file = clicks_file
        self.buys_dict_file = buys_file
        self.parameters = {}


    def match_clicks_buys( self, clicks, buys ):
        """ given a list of clicks and buys in the
        same session, return the corresponding seq:
            clicks_seq = [ x1, x2, ..., xn ]
            buys_seq   = [ b1, b2, ..., bn ]
        where xi ~ item_id and bi ~ [0,1] meaning
        not buy/ buy.
        """
        len_clicks = len( clicks )
        clicks_seq, buys_seq = [], []
        for click_index in range( len_clicks ):
            click_record = clicks[ click_index ]
            item = click_record[ 'item' ]

            # judge if this item is bought
            buy = 0
            temp_index = click_index + 1
            for buy_record in buys:
                if buy_record[ 'item' ] == item \
                and buy_record[ 'time' ] > click_record[ 'time' ]:
                    buy = 1
                    while temp_index < len_clicks:
                        temp_click = clicks[ temp_index ]
                        if temp_click[ 'item' ] == item \
                        and buy_record[ 'time' ] > temp_click[ 'time' ]:
                            buy = 0
                        temp_index += 1
                if buy == 1 : break

            # update sequences
            clicks_seq.append( item )
            buys_seq.append( buy )

        return clicks_seq, buys_seq

    def generate_params( self, clicks_seq, buys_seq ):
        """ Generate parameters for sequence inference:
            P( Xi ) = [ n( bi=0 ), n( bi=1 ) ]
            P( Xi-1,Xi ) = [ n( bi-1=0, bi=0 ) , n( bi-1=0, bi=1) ,
                             n( bi-1=1, bi=0 ) , n( bi-1=1, bi=1) ]
        """
        assert len( clicks_seq ) == len( buys_seq ), \
                "The clicks_seq and buys_seq are not of same length!"
        len_seq = len( clicks_seq )
        params_dict = {}
        for i in range( len_seq ):
            # update params for current item
            item = clicks_seq[ i ]
            buy = buys_seq[ i ]
            params_dict.setdefault( item, \
                                    numpy.array([ 0, 0 ]) )
            params_dict[ item ][ buy ] += 1

            # update params for item together with next item
            if i < ( len_seq - 1 ):
                item_next = clicks_seq[ i+1 ]
                buy_next = buys_seq[ i+1 ]
                pair = item + '_' + item_next
                params_dict.setdefault( pair, \
                                        numpy.array([ [0, 0],
                                                      [0, 0] ])
                                        )
                params_dict[ pair ][ buy ][ buy_next ] += 1

        return params_dict


    def merge_params( self, new_params ):
        """ Merge the new generated parameters to the
        global parameters set
        """
        assert type( new_params ) is dict, 'New_params is not dict type'
        for key in new_params:
            if key in self.parameters:
                self.parameters[ key ] += new_params[ key ]
            else:
                self.parameters[ key ] = new_params[ key ]


    def unify_params( self ):
        """ Refresh parameters to avoid pick point divided by 0 
        """
        for key in self.parameters:
            if '_' not in key:
                self.parameters[ key ] += numpy.array([1, 1])
            else:
                self.parameters[ key ] += numpy.array(
                                    [ [1, 1],
                                      [1, 1] ]
                                  )

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

            clicks, buys = self.match_clicks_buys( clicks, buys )

            new_params = self.generate_params( clicks, buys )
            self.merge_params( new_params )
        self.unify_params()
        print "\n\t Parameters generation finished~"


    def store_params( self, dir_2_store ):
        """ Store parameters dict to the directory for later use
        """
        print "\n\t Start storing parameters"
        with open( dir_2_store + os.sep + 'sequence_params.dict' , 'wb' ) as fstream:
            cPickle.dump( self.parameters, fstream, -1 )
        print "\n\t Parameters storage finished~"


class SequencePrediction:

    def __init__( self, params_file, test_file ):
        """ Initiate class with params/test file path
        and the initial probability distribution on
        single and neighbor estimator.
        """
        self.params_file = params_file
        self.test_file = test_file
        self.init_proportion = { 'single': 0.3,
                                 'pair': 0.7 }
        self.results = {}

    def load_params( self ):
        print "\t Start Load parameters from sequence model"
        with open( self.params_file, 'rb' ) as fstream:
            self.parameters = cPickle.load( fstream )
        print "\t Load parameters Finished ~~"

    def predict( self, clicks ):
        """ Given a series of clicks in one session
        generate the most probable buys sequence
            B[ b1, b2, ..., bn ]
        where bi~[0, 1]
        """
        len_clicks = len( clicks )
        buys = []
        for i in range( len_clicks ):
            item = clicks[ i ]
            item_params = self.parameters.get( item,
                                               numpy.array([ 1, 1])
                                              )
            if i == 0 :
                if item_params[0] >= item_params[1]:
                    buys.append( 0 )
                else:
                    buys.append( 1 )
                continue

            last_item = clicks[ i-1 ]
            last_buy = buys[-1]
            pair = last_item + '_' + item
            pair_params = self.parameters.get( pair,
                                                numpy.array(
                                                    [[ 1, 1 ],
                                                     [ 1, 1 ]]
                                                            )
                                             )
            prob_buy = self.init_proportion[ 'single' ] \
                        * item_params[1] / sum( item_params ) \
                      + self.init_proportion[ 'pair' ] \
                         * pair_params[ last_buy ][1] / sum( pair_params[ last_buy ])

            prob_not_buy = self.init_proportion[ 'single' ] \
                            * item_params[0] / sum( item_params ) \
                          + self.init_proportion[ 'pair' ] \
                             * pair_params[ last_buy ][0] / sum( pair_params[ last_buy ])

            if prob_not_buy >= prob_buy:
                buys.append( 0 )
            else:
                buys.append( 1 )

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
            clicks_seq = [ click[ 'item' ] for click in clicks ]
            buys_seq = self.predict( clicks_seq )
            if 1 in buys_seq:
                self.results[ session ] = []
                for i in range( len( buys_seq ) ):
                    if buys_seq[ i ] == 1:
                        self.results[ session ].append( clicks_seq[ i ] )

        #Store prediction results to a csv
        print "\t Start storing results to ", dir_2_store
        with open( dir_2_store + os.sep + 'seq_results' , 'w' ) as fstream:
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
    dir_2_store = "../median_datasets"
    dir_2_store = "../results_datasets"
    """
    # generate parameters and store them
    sequence_model = SequenceModelCreation( clicks_file, buys_file )
    sequence_model.create()
    sequence_model.store_params( dir_2_store )
    """
    """
    # use model to predict results
    prediction = SequencePrediction( "../median_datasets/sequence_params.dict", clicks_file )
    prediction.load_params()
    prediction.do_task( dir_2_store )
    """
    # calculate score
    result_file = "../results_datasets/seq_results"
    evaluation = ResultsEvaluation( clicks_file, result_file, buys_file)
    #evaluation.cal_score()
    evaluation.cal_precision()


if __name__ == "__main__":
    main()
