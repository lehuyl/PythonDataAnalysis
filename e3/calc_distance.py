import sys
import numpy as np
import pandas as pd
from pykalman import KalmanFilter


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


def get_data(file):
    import xml.dom.minidom
    document = xml.dom.minidom.parse(file);
    elements = document.getElementsByTagName("trkpt")
    
    
    coord = []
    for element in elements:
        coord.append({'lat': element.getAttribute('lat'),'lon': element.getAttribute('lon')})  
    df = pd.DataFrame(coord)
    df = df.apply(pd.to_numeric)
    
    return df

# adapted from http://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 # Radius of the earth in m
    dLat = np.deg2rad(lat2-lat1);  # deg2rad below
    dLon = np.deg2rad(lon2-lon1); 
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)); 
    d = R * c; # Distance in m
    
    return d;
    
#print(haversine(49.28, 123.00, 49.26, 123.1) + haversine(49.26,123.1,49.26,123.05))
# https://stackoverflow.com/questions/23142967/adding-a-column-thats-result-of-difference-in-consecutive-rows-in-pandas
def distance(points):
    #points.astype(float)
    
    points['vector']= np.vectorize(haversine)(points['lat'], points['lon'], points['lat'].shift(-1), points['lon'].shift(-1))
    
    return np.sum(points['vector'])
    
def smooth(points):
    kalman_data = points[['lat','lon']]
    initial_state = kalman_data.iloc[0]
    observation_covariance = np.diag([7.4, 6]) ** 2 # TODO: shouldn't be zero
    transition_covariance = np.diag([2, 2]) ** 2 # TODO: shouldn't be zero
    transition = [[1, 0], [0, 1]]
    
    kf = KalmanFilter(
        initial_state_mean=initial_state,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition)
    
    kalman_smoothed, _ = kf.smooth(kalman_data)
    df = pd.DataFrame(kalman_smoothed)
    df.columns = ['lat','lon']
    
    return df

    
    

def main():
    points = get_data(sys.argv[1])
    print('Unfiltered distance: %0.2f' % (distance(points),))
    
    smoothed_points = smooth(points)
    print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    output_gpx(smoothed_points, 'out.gpx')


if __name__ == '__main__':
    main()