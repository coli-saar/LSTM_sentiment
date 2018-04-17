Data was produced as follows:

sed -n -e "1,800000 p" -e "800000 q" review.json > review1M-train.json
sed -n -e "800001,900000 p" -e "900000 q" review.json > review1M-dev.json
sed -n -e "900001,1000000 p" -e "1000000 q" review.json > review1M-test.json

