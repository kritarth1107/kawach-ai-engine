# Engine owners

One owner reviews every change in an area. A commit that touches someone else's area carries
`Reviewed-by: <owner>`. Every commit carries `Author-agent: <name>`. `scripts/owners_check.py` reads the table below.
The nightly trainer (`Author-agent: Saheli Trainer`) may edit `app/brain/persona/` without the owner's trailer: the founder
reviews and merges its pull request. Merge commits are not checked; their commits are.

```owners
app/brain/loop.py	Brain Engineer
app/brain/context.py	Brain Engineer
app/brain/persona/	Brain Engineer
app/llm/	Brain Engineer
app/care/store.py	Brain Engineer
app/care/extract.py	Brain Engineer
app/care/dream.py	Brain Engineer
app/care/memory_index.py	Brain Engineer
app/care/versions.py	Brain Engineer
app/brain/checks/	Saheli Quality Engineer
app/brain/guards.py	Saheli Quality Engineer
app/quality/	Saheli Quality Engineer
app/learn/	Saheli Quality Engineer
eval/	Saheli Quality Engineer
app/brain/tools.py	Kuchhu Puchhu
app/brain/tools/__init__.py	Kuchhu Puchhu
app/tasks/	Order Engineer
app/brain/tools/orders.py	Order Engineer
app/brain/tools/trips.py	Order Engineer
app/nutrition/	Nutrition Engineer
app/brain/tools/food.py	Nutrition Engineer
app/care/nudges.py	Kavach NUdge
app/care/nudges/	Kavach NUdge
app/care/checkins.py	Kavach NUdge
app/care/activities.py	Kavach NUdge
app/brain/tools/reminders.py	Kavach NUdge
.github/	Reliability Engineer
app/jobs/	Reliability Engineer
scripts/owners_check.py	Reliability Engineer
scripts/saheli_check.sh	Reliability Engineer
OWNERS.md	Reliability Engineer
```
