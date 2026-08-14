# Tenant isolation

**Rule: Family A must never see Family B's chat, labs, or RAG context.**

## How it is enforced

### 1. Every row is scoped by `family_id`

All memory tables include `family_id` as a foreign key:

- `messages`
- `memory_snippets`
- `document_chunks`
- `memory_documents`
- `conversations`
- `elders`

There is **no shared memory table** across families.

### 2. Every API call validates ownership first

Before chat or document operations, `app/services/tenant.py` checks:

| Check | Function |
|-------|----------|
| Family exists | `assert_family_exists` |
| Elder belongs to family | `assert_elder_in_family` |
| Conversation belongs to family + elder | `assert_conversation_in_family` |

If Family B passes Family A's `elder_id` or `conversation_id`, the API returns **404** — no data leaks.

### 3. RAG queries always filter by `family_id`

```sql
WHERE family_id = :family_id
  AND (elder_id IS NULL OR elder_id = :elder_id)  -- same family only
```

Vector search never runs across families. Embeddings from Family A cannot appear in Family B's results.

### 4. Cross-chat memory is **within one family only**

"Cross-chat" means: all Telegram + in-app threads **for the same elder in the same family**.

It does **not** mean across families.

### 5. LangGraph state carries `family_id`

The Saheli graph receives `family_id` + `elder_id` on every invoke. Retrieve and persist nodes use only those IDs.

## What the Next.js app must do

When calling this backend:

1. Resolve the signed-in user's family via `family_members` (never trust client-supplied family id without membership check)
2. Pass that `family_id` on every `/v1/chat` and `/v1/documents/*` request
3. Never reuse conversation IDs across families

## Checklist for new features

- [ ] Query includes `WHERE family_id = ?`
- [ ] API validates elder/conversation belong to family
- [ ] No global cache keys without family prefix
- [ ] No shared vector index across tenants
