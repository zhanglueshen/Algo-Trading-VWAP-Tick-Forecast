import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import sys

from simtools import log_message

# VWAP with factor affecting theoretical value

matplotlib.rcParams[ 'figure.figsize' ] = ( 14, 6 )

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# use coefs to make our target series
def vwap_target( bar_num, coefs ):
    return ( coefs[ 0 ] * bar_num +
             coefs[ 1 ] * bar_num**2 +
             coefs[ 2 ] * bar_num**3 +
             coefs[ 3 ] * bar_num**4 +
             coefs[ 4 ] * bar_num**5 )

# Record a trade in our trade array
def record_trade( trade_df, idx, trade_px, trade_qty, current_bar, trade_type ):
    #print( "Trade! {} {} {} {}".format( idx, trade_px, trade_qty, current_bar ) )
    trade_df.loc[ idx ] = [ trade_px, trade_qty, current_bar, trade_type ]

    return

# Get next order quantity
def calc_order_quantity( raw_order_qty, round_lot ):
    if raw_order_qty >= round_lot: # round to nearest lot
        return np.around( int( raw_order_qty ), int( -1 * np.log10( round_lot ) ) )
    # our target quantity is small... are we almost done?
    elif raw_order_qty > 0:
        # we're less than 1 lot from completion, set for full size
        return raw_order_qty 
    # we shouldn't get here. If we do, something is weird
    else:
        return -1


# calc the schedule factor 
# behind maximum value = 1, ahead maximum value = -1
def calc_schedule_factor( max_behind, target_shares, quantity_filled, order_quantity ):
    # calculate tolerance factor in spread units in range [-0.5..0.5]
    quantity_behind = target_shares - quantity_filled
    
    # scale tolerance to half a spread in $
    schedule_factor = (quantity_behind / max_behind) 
    
    # cap the schedule factor at twice
    schedule_factor = min( 1, max( -1, schedule_factor ) )

    return schedule_factor

    
# MAIN ALGO LOOP
def algo_loop( trading_day, order_side, original_order_quantity, vwap_coefficients, max_behind, schedule_coef):
    log_message( f"Beginning VWAP run:{order_side},{original_order_quantity} shares, max_behind:{max_behind}, schedule_coef: {schedule_coef}")

    round_lot = 100
    avg_spread = ( trading_day.ask_px - trading_day.bid_px ).mean()
    print( "Average stock spread for sample: {:.4f}".format(avg_spread) )

    # generate target schedule - use bins 1 - 390 giving an automatic 1 minute "look ahead"
    # note that targets have been converted to shares from percent
    order_targets = vwap_target( np.arange( 0, 390, dtype='int64'), vwap_coefficients ) * original_order_quantity
    
    # let's force the last bin to complete
    order_targets[-1] = original_order_quantity
    
    # generate last-minute bar
    last_time_from_open = (trading_day.index[-1] - pd.Timedelta( hours = 9, minutes = 30 ))
    last_minute_from_open = (last_time_from_open.hour * 60) + last_time_from_open.minute

    # init our price and volume variables
    [ last_price, last_size, bid_price, bid_size, ask_price, ask_size, volume ] = np.zeros(7)

    # init our counters
    [ trade_count, quote_count, cumulative_volume ] = [ 0, 0, 0 ]
    
    # init some time series objects for collection of telemetry
    fair_values = pd.Series( index=trading_day.index )
    midpoints = pd.Series( index=trading_day.index )
    schedule_factors = pd.Series( index=trading_day.index )
    
    # let's set up a container to hold trades. preinitialize with the index
    trades = pd.DataFrame( columns = [ 'price' , 'shares', 'bar', 'trade_type' ], index=trading_day.index )

    # MAIN EVENT LOOP
    current_bar = 0
    current_target_shares = 0

    quantity_behind = 0

    # track state and values for a current working order
    live_order = False
    live_order_price = 0.0
    live_order_quantity = 0.0

    # other order and market variables
    total_quantity_filled = 0
    #quantity_remaining = original_order_quantity - total_quantity_filled
    vwap_numerator = 0.0

    total_trade_count = 0
    total_agg_count = 0
    total_pass_count = 0

    # fair value pricing variables
    midpoint = 0.0
    fair_value = 0.0
    schedule_factor = 0.0
    if (order_side=='s') & (schedule_coef>=0):
        schedule_coef = -schedule_coef # change the schedule factor sign for sell orders
    
    log_message( 'starting main loop' )
    for index, row in trading_day.iterrows():
        # get the time of this message
        time_from_open = (index - pd.Timedelta( hours = 9, minutes = 30 ))
        minutes_from_open = (time_from_open.hour * 60) + time_from_open.minute
        
        # MARKET DATA HANDLING
        if pd.isna( row.trade_px ): # it's a quote
            # skip if not NBBO
            #if not ( ( row.qu_source == 'N' ) and ( row.natbbo_ind == 4 ) ):
            #    continue
            # set our local NBBO variables
            if ( row.bid_px > 0 and row.bid_size > 0 ):
                bid_price = row.bid_px
                bid_size = row.bid_size * round_lot
            if ( row.ask_px > 0 and row.ask_size > 0 ):
                ask_price = row.ask_px
                ask_size = row.ask_size * round_lot
            quote_count += 1
        else: # it's a trade
            last_price = row.trade_px
            last_size = row.trade_size
            trade_count += 1
            cumulative_volume += row.trade_size
            vwap_numerator += last_size * last_price

            # CHECK OPEN ORDER(S) if we have a live order, 
            # has it been filled by the trade that just happened?
            if live_order :
                if ( order_side == 'b' ) and ( last_price <= live_order_price ) :
                    fill_size = min( live_order_quantity, last_size )
                    record_trade( trades, index, live_order_price, fill_size, current_bar, 'p' )
                    total_quantity_filled += fill_size
                    total_pass_count += 1

                    # even if we only got partially filled, let's assume we're cancelling the entire quantity. 
                    # If we're still behind we'll replace later in the loop
                    live_order = False
                    live_order_price = 0.0
                    live_order_quantity = 0.0
                    quantity_behind = current_target_shares - total_quantity_filled

                if ( order_side == 's' ) and ( last_price >= live_order_price ) :
                    fill_size = min( live_order_quantity, last_size )
                    record_trade( trades, index, live_order_price, fill_size, current_bar, 'p' )
                    total_quantity_filled += fill_size
                    total_pass_count += 1

                    # even if we only got partially filled, let's assume we're cancelling the entire quantity. 
                    # If we're still behind we'll replace later in the loop
                    live_order = False
                    live_order_price = 0.0
                    live_order_quantity = 0.0
                    quantity_behind = current_target_shares - total_quantity_filled

        # SCHEDULE FACTOR
        if minutes_from_open > current_bar:
            # we're in a new bar do new bar things here
            current_bar = minutes_from_open
            current_target_shares = order_targets[ current_bar ]
            quantity_behind = current_target_shares - total_quantity_filled
            # uncomment for debug
            #print( "{} bar: {:d} target:{:.2f}".format(index, current_bar, current_target_shares) )

        # PRICING LOGIC
        new_midpoint = bid_price + ( ask_price - bid_price ) / 2
        if new_midpoint > 0:
            midpoint = new_midpoint

        schedule_factor = calc_schedule_factor( max_behind, current_target_shares, 
                                               total_quantity_filled, 
                                               original_order_quantity )

        # FAIR VALUE CALCULATION
        # check inputs, skip of the midpoint is zero, we've got bogus data (or we're at start of day)
        if midpoint == 0:
            #print( "{} no midpoint. b:{} a:{}".format( index, bid_price, ask_price ) )
            continue
        fair_value = midpoint + (schedule_coef * schedule_factor * avg_spread / 2)

        # collect our data
        fair_values[index] = fair_value
        midpoints[index] = midpoint
        schedule_factors[index] = schedule_factor
        
        # TRADING LOGIC
        # check where our FV is versus the BBO and constrain
        if order_side == 'b':
            if (fair_value >= ask_price and quantity_behind > round_lot) or (minutes_from_open==last_minute_from_open and quantity_behind > 0): 
                total_agg_count += 1

                new_trade_price = ask_price

                # now place our aggressive order: assume you can execute the full size across spread
                new_order_quantity = calc_order_quantity( quantity_behind, round_lot )
                
                record_trade( trades, index, new_trade_price, new_order_quantity, current_bar, 'a' )

                # update total quantity filled
                total_quantity_filled += new_order_quantity
                
                # update quantity behind
                quantity_behind = current_target_shares - total_quantity_filled
                
                live_order_quantity = 0.0
                live_order_price = 0.0
                live_order = False

            else: # we're not yet willing to cross the spread, stay passive
                if quantity_behind > round_lot:
                    live_order_price = bid_price
                    live_order_quantity = calc_order_quantity( quantity_behind, round_lot)
                    #live_order_quantity = quantity_behind
                    live_order = True

        elif order_side == 's':
            if (fair_value <= bid_price and quantity_behind > round_lot) or (minutes_from_open==last_minute_from_open and quantity_behind > 0):
                total_agg_count += 1

                new_trade_price = bid_price

                # now place our aggressive order: assume you can execute the full size across spread
                new_order_quantity = calc_order_quantity( quantity_behind, round_lot) 
                
                #new_order_quantity = quantity_behind
                record_trade( trades, index, new_trade_price, new_order_quantity, current_bar, 'a' )

                # update total quantity filled
                total_quantity_filled += new_order_quantity
                
                # update quantity behind
                quantity_behind = current_target_shares - total_quantity_filled

                live_order_quantity = 0.0
                live_order_price = 0.0
                live_order = False

            else: # not yet willing to cross spread
                if quantity_behind > round_lot:
                    live_order_price = ask_price 
                    live_order_quantity = calc_order_quantity( quantity_behind, round_lot)
                    #live_order_quantity = quantity_behind
                    live_order = True
        else:
            # we shouldn't have got here, something is wrong with our order type
            print( 'Got an unexpected order_side value: ' + str( order_side ) )

    # looping done
    log_message( 'end simulation loop' )
    log_message( 'order analytics' )

    # Now, let's look at some stats
    trades = trades.dropna()
    day_vwap = vwap_numerator / cumulative_volume

    # prep our text output
    avg_price = (trades[ 'price' ] * trades[ 'shares' ]).sum() / trades[ 'shares' ].sum()

    log_message( 'VWAP run complete.' )

    # assemble results and return
    
    return { 'midpoints' : midpoints,
             'fair_values' : fair_values,
             'schedule_factors' : schedule_factors,
             'trades' : trades,
             'quote_count' : quote_count,
             'day_vwap' : day_vwap,
             'avg_price' : avg_price,
             'order_targets' : order_targets
           }