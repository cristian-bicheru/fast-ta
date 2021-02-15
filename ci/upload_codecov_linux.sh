curl --connect-timeout 10 --max-time 10 --retry 5 --retry-delay 0 --retry-max-time 120 https://codecov.io/bash -o uploader.sh
chmod +x uploader.sh
./uploader.sh
