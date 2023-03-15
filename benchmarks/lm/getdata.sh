# Copyright (c) 2023  Khaleelulla Khan Nazeer, Anand Subramoney, Mark Schöne, David Kappel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# -------------------------------------
# Usage: ./getdata.sh <data_directory>
# -------------------------------------
echo "=== Acquiring datasets ==="
echo "---"

cd $1

echo "- Downloading WikiText-2 (WT2)"
wget --quiet --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip -q wikitext-2-v1.zip
cd wikitext-2
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt
cd ..

echo "- Downloading WikiText-103 (WT2)"
wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip -q wikitext-103-v1.zip
cd wikitext-103
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt
cd ..

echo "- Downloading Penn Treebank (PTB)"
wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf simple-examples.tgz

mkdir -p PennTreebank
cd PennTreebank
mv ../simple-examples/data/ptb.train.txt train.txt
mv ../simple-examples/data/ptb.test.txt test.txt
mv ../simple-examples/data/ptb.valid.txt valid.txt
cd ..

echo "---"
echo "Happy language modeling :)"
