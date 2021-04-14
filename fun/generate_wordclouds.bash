#!/usr/bin/env bash

set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

current_date="$1"
end="$2"
# https://stackoverflow.com/a/25701378
while ! [[ $current_dated > $end ]]; do
    ./generate_wordcloud.py $current_date
    current_date=$(gdate -d "$current_date + 1 day" +%F)
done