# generate tests
python3 -m interface.generate_test --db 20 --f 11500 --t 6 --downsample True
# create the fingerprints
python3 -m interface.create_fingerprints
# run analyze
python3 -m interface.analyze

# delete 20
python3 -m interface.delete

# generate tests
python3 -m interface.generate_test --db 20 --f 11500 --t 6 
# create the fingerprints
python3 -m interface.create_fingerprints
# run analyze
python3 -m interface.analyze

# delete 20
python3 -m interface.delete

# generate tests
python3 -m interface.generate_test --db 20 --f 11500 --t 6 
# create the fingerprints
python3 -m interface.create_fingerprints
# run analyze
python3 -m interface.analyze

