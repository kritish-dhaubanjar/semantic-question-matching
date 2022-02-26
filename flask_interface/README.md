## Deployment Setup with Docker (Development)

1. Download [models](https://drive.google.com/open?id=1YgibRxIBRPDBvrPPstxkInnNKc6M5lFc), archive and extract.

2. Download [googlenews-vectors-negative300.bin.gz](https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz).

3. Copy all models into `models` directory.

   - ANN.model
   - KNN.model
   - word2vec.model
   - RandomForest.model
   - LogisticRegression.model
   - GoogleNews-vectors-negative300.bin.gz

4. Build and run application with `docker-compose up`.

![docker-compose_build](https://user-images.githubusercontent.com/25634165/155836579-be0ba358-941a-4409-91d2-80920003d7f5.png)
