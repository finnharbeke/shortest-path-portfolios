# shortest-path-portfolios
aiming to find baselines and new methods for the problem of portfolio optimization for graph's shortest paths using stochastic information

## game plan

1. unified graph class
   - parsing the dataset(s)
2. picking pairs
3. baselines
4. heuristics
    - kmeans
    - gnns

## data

sources:

shenzhen & los angeles: https://github.com/lehaifeng/T-GCN
shanghai: https://github.com/xxArbiter/grnn

### shenzhen

> *SZ-taxi*. This dataset was the taxi trajectory of Shenzhen from Jan. 1 to Jan. 31, 2015. We selected 156 major roads of Luohu District as the study area. The experimental data mainly includes two parts. One is an 156*156 adjacency matrix, which describes the spatial relationship between roads. Each row represents one road and the values in the matrix represent the connectivity between the roads. Another one is a feature matrix, which describes the speed changes over time on each road. Each row represents one road; each column is the traffic speed on the roads in different time periods. We aggregate the traffic speed on each road every 15 minutes.

these are the `data/sz_*.csv` files.

*binary adjancency matrix and traffic speed*

### los angeles

> Los-loop. This dataset was collected in the highway of Los Angeles County in real time by loop detectors. We selected 207 sensors and its traffic speed from Mar.1 to Mar.7, 2012. We aggregated the traffic speed every 5 minutes. Similarity, the data concludes an adjacency matrix and a feature matrix. The adjacency matrix is calculated by the distance between sensors in the traffic networks. Since the Los-loop dataset contained some missing data, we used the linear interpolation method to fill missing values.

these are the `data/la_*` files.

*weighted adjancency matrix and traffic speed*

### shanghai

> Raw taxi trajectory dataset we use in this research is obtained from TIC Shanghai, the distribution of samplings are illustrated in Figure 7. To be specific, 310 GB data are gathered from 13, 573 taxis from Apr. 1, 2015 to Apr. 30, 2015 and a city-scale road network contains 65,836 road segments. Each taxi reports the GPS report every 10 seconds. The raw trajectory data include the ID, geographical position, upload time stamp, carrying state, speed, the ori- entation of the vehicle and so on. We mined and restored the traffic conditions of all segments in that time span in our previous work (Wang et al. 2018), and set the time interval to 10 minutes. Unfortunately, samples from most of the seg- ments are too sparse, in other words, we only have a set of segments with entire time series of traffic conditions. Thus, we select a connected subgraph with 156 vertexes as shown in the attached graph on the right side of Figure 7 with highest sampling density as our test bed where all following ex- periments will be executed. To be noticed, all the raw data are private, but the processed testbed is available on GitHub, together with the codes of the proposed scheme and tests: https://github.com/xxArbiter/grnn.

*proper lat lon of start & end per segment*
