 # coding: utf-8


import sys
import os
import re
import random
import time
import SekiteiExtractFeatures as ef 
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
from sklearn.ensemble import GradientBoostingClassifier
import numpy
import urllib

#clustering class
class StackClustering:

      def __init__(self, min_distance=0.6):
      
          self.__min_distance = min_distance;
      
      def fit_predict(self, data):
    	  data_length = len(data);
    	  marks = [-1] * data_length
    	  clusters = 0
    	  self.cluster_features = {};
          for i, (v1) in enumerate(data):
              if marks[i] != -1:
                 continue;
              marks[i] = clusters;
              if clusters not in self.cluster_features:
                 self.cluster_features[clusters] = v1;
              for j in range(i+1, data_length):
                  if marks[j] != -1:
                     continue;
                  v2 = data[j]
                  
                  dist =  1 - cosine(v1, v2)
                  if dist > self.__min_distance:
                     marks[j] = clusters;
                     
                        
              clusters += 1;
          return marks;
      
      def predict(self, X):
          result = [];
          
          inter = 0;
          cond  = 0;
          the_best = (0, 0);
          for k in  self.cluster_features.keys():
               v2 = self.cluster_features[k];
               dist = 1 - cosine(v2, X)
               if dist > the_best[0]:
                  the_best = (dist, k)
          result.append(the_best);
          return result; 
            
                  


#
# class performs urls segmentation
#
class SekiteiUrlsSegmentation:
    
    __MIN_QUOTA = 100;
    
    def __init__(self, QLINKS_URLS, UNKNOWN_URLS, 
                       ALFA = 0.01, SEGMENTS=60, MAX_QUOTA=10000):
        self.ALFA = ALFA
        self.MAX_QUOTA = MAX_QUOTA
        self.segments_count = SEGMENTS;
        self.extract_features(QLINKS_URLS, UNKNOWN_URLS);
        
        self.__clusters = KMeans(SEGMENTS) #StackClustering(); # KMeans(SEGMENTS);
        self.classifier_algoritm = GradientBoostingClassifier(learning_rate=0.7, n_estimators=100)
        
        self.__clustering(QLINKS_URLS, UNKNOWN_URLS)
        self.__trash_segment = 0;
        self.__total_fetched = 0;
        
    # performs urls clustering
    def __clustering(self, QLINKS_URLS, UNKNOWN_URLS):
        urls = [];
        
        q_vector_set = self.urls_to_features(QLINKS_URLS, 1);
        n_vector_set = self.urls_to_features(UNKNOWN_URLS, 0);
        
        urls.extend(q_vector_set);
        urls.extend(n_vector_set);
        data = None; 
        flen =  0;
        for u in urls:
            a = numpy.asarray(u[0], numpy.float32);
            flen = len(u[0]);
            if data is None:
                data = a;
            else:
                data = numpy.vstack((data, a))
                
        self.__segments = { };        
        self.__segments_qlink_weight = {};
        predictions = self.__clusters.fit_predict(data)
        #score = self.__clusters.score(data);
        qsum = 0.0
        segments_data = {}
        urls_by_clusters = {}
        for idx, (seg_idx) in enumerate(predictions):
            if seg_idx not in self.__segments:
                self.__segments[seg_idx] = 0;
                self.__segments_qlink_weight[seg_idx] = 0
                segments_data[seg_idx] = data[idx];
                urls_by_clusters[seg_idx] = []
            else:
                segments_data[seg_idx] = numpy.vstack((segments_data[seg_idx], data[idx]))
            
            urls_by_clusters[seg_idx].append((urls[idx][1],urls[idx][2] ))
                
            self.__segments[seg_idx] += float(urls[idx][2]);
            self.__segments_qlink_weight[seg_idx] += 1;
            
            qsum +=  urls[idx][2];
            	      
        quota_sum = 0;
        max_quota = 0
        for seg in self.__segments.keys():
            qpart = self.__segments[seg] / len(urls);
            if qpart > max_quota:
               max_quota = qpart;
               
        self.__min_quota = self.__MIN_QUOTA
        if max_quota > 0.1:
           self.__min_quota = 2 * self.__min_quota;
           
        for seg in self.__segments.keys():
            self.__segments_qlink_weight[seg] = self.__segments[seg] / self.__segments_qlink_weight[seg] 
            self.__segments[seg] =  self.MAX_QUOTA * self.__segments[seg] / qsum + self.__min_quota
            quota_sum += self.__segments[seg]
            
        self.__seg_quotes = [0.] * self.segments_count;
        Y = numpy.hstack((numpy.ones(len(q_vector_set), int), numpy.zeros(len(n_vector_set), int)))
        self.classifier_algoritm.fit(data, Y)

# fetching url         
    def fetch_url(self, url):
        X = numpy.asarray(self.url_to_features(url,0)[0][0]);
        num = self.__clusters.predict(X)[0];
        if num in self.__segments: 

	   if self.__segments[num] > self.__seg_quotes[num]:
	        predict = True; 
	        if self.__segments[num] > self.__min_quota:
            	    predict =  self.classifier_algoritm.predict(X)[0]
           	    if predict == False and random.random() < self.__segments_qlink_weight[num]: predict = True; 
                if predict :
            	    self.__seg_quotes[num] += 1;
            	    self.__total_fetched += 1;
            	    return True;                       
               
        return False;
        
        
    def sample_urls_from_file(self, FILE_NAME, count):
        urls = [];
        with open(FILE_NAME ) as i_file:
             for line in i_file:
                line = line.strip();
                urls.append(line);
        random.shuffle(urls)
        return urls [:count];  
    
    def extract_features(self, QLINKS_URLS, UNKNOWN_URLS):
    
        urls = []

        urls.extend(QLINKS_URLS);
        urls.extend(UNKNOWN_URLS);
        
        extractor = ef.SekiteiFeatureExtractor(urls) 
        extractor.extract_features();
        self.__stat = extractor.calculate_stat(self.ALFA);
        self.__vsize = len(self.__stat)
        self.__feas = {}; # map features to index
        self.__idx2fea = {}; # map feature index to fea name
        for fidx, (fname, fstat) in enumerate(self.__stat):
            self.__feas[fname] = fidx;
            self.__idx2fea[fidx] = fname;
            
    def get_fea_names(self):
        names = [""] * self.__vsize;
        for fname in self.__feas.keys():
            names[self.__feas[fname]] = fname;
        return names;
    
    # convert single url to features
    def url_to_features(self, url, urls_class):
        urls = []
        vector_set = [];
        urls.append(url);
        extractor = ef.SekiteiFeatureExtractor(urls)
        extractor.extract_features();
        feas, urls = extractor.features();
        
        for i, (urlfs) in enumerate(feas):
            vector = [0.] * self.__vsize;
            for fname in urlfs:
                if fname in self.__feas:
                    vector[self.__feas[fname]]  = 1. # this veature is present in url
            vector_set.append((vector,  urls[i], urls_class));
        
        return vector_set
    
    def urls_to_features(self, urls, urls_class):
        vector_set = [];
        extractor = ef.SekiteiFeatureExtractor(urls)
        extractor.extract_features();
        feas, urls = extractor.features();
        
        for i, (urlfs) in enumerate(feas):
            vector = [0.] * self.__vsize;
            for fname in urlfs:
                if fname in self.__feas:
                    vector[self.__feas[fname]]  = 1. # this veature is present in url
            vector_set.append((vector,  urls[i], urls_class));
        return vector_set;
    
    
    def urls_to_features_from_file(self, INPUT_FILE, urls_class):
        urls = []
        with open(INPUT_FILE ) as i_file:
             for line in i_file:
                line = line.strip();
                urls.append(line);
                
        return self.urls_to_features(urls, urls_class);
    
    def url_to_segment(url):
        ("")
        
    def __to_som_vec_file(self, OUTPUT_SOM_BASE, vector_set):
        with open(OUTPUT_SOM_BASE + ".vec", "w") as vec_file:
            print >> vec_file, "$XDIM " + str(len(vector_set))
            print >> vec_file, "$YDIM 1"
            print >> vec_file, "$VEC_DIM " + str(self.__vsize)
            for vector,url, cl in vector_set:
                url = urllib.quote(url);
                print >> vec_file, ' '.join(format(v, ".3f") for v in vector) + " "+ url;
                
    def __to_som_tv_file(self, OUTPUT_SOM_BASE):
        with open(OUTPUT_SOM_BASE + ".tv", "w") as tv_file:
            print >> tv_file, "$TYPE template";
            print >> tv_file, "$XDIM 2";
            print >> tv_file, "$YDIM 1";
            print >> tv_file, "$VEC_DIM " + str(len(self.__feas))
            fnames = self.get_fea_names();
            for idx, (name) in enumerate(fnames):
                print >> tv_file, str(idx) + " " + name;
            
    def __to_som_class_file(self, OUTPUT_SOM_BASE, vector_set):
        with open(OUTPUT_SOM_BASE + ".cls", "w") as class_file:
            for vector,url, cl in vector_set:
                url = urllib.quote(url);
                print >> class_file, url + "\t" + str(cl)
            
    def to_som_input_files(self, vector_set, OUTPUT_SOM_BASE):
        self.__to_som_vec_file(OUTPUT_SOM_BASE, vector_set);
        self.__to_som_tv_file(OUTPUT_SOM_BASE)
        self.__to_som_class_file(OUTPUT_SOM_BASE, vector_set);

sekitei = None;        
        
def define_segments(QLINK_URLS, UNKNOWN_URLS, QUOTA):
    global sekitei
    sekitei = SekiteiUrlsSegmentation(QLINK_URLS, UNKNOWN_URLS, MAX_QUOTA=QUOTA);
    

# returns segment    
def fetch_url(url):
    global sekitei
    return sekitei.fetch_url(url);
