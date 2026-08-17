package com.anisync.android.data.ai

import com.anisync.android.domain.ai.AiGroundingSource
import com.anisync.android.domain.ai.AiMediaFocusContext
import com.anisync.android.domain.ai.AiUserDataEntry
import com.anisync.android.domain.ai.ChatMessage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class GeminiApiService @Inject constructor(
    private val baseOkHttpClient: OkHttpClient
) {
    private val client: OkHttpClient by lazy {
        baseOkHttpClient.newBuilder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()
    }

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
    }

    data class GenerateResult(
        val text: String,
        val sources: List<AiGroundingSource> = emptyList(),
        val thinkingProcess: String? = null
    )

    suspend fun generateChatResponse(
        apiKey: String,
        modelName: String = "gemini-2.5-flash",
        conversationHistory: List<ChatMessage>,
        latestUserMessage: String,
        useWebSearch: Boolean = true,
        allowSpoilers: Boolean = false,
        userData: List<AiUserDataEntry> = emptyList(),
        mediaFocus: AiMediaFocusContext? = null
    ): GenerateResult = withContext(Dispatchers.IO) {
        if (apiKey.isBlank()) {
            throw IllegalArgumentException("Please enter your Google Gemini API key in Settings -> AI Assistant.")
        }

        val endpoint = "https://generativelanguage.googleapis.com/v1beta/models/$modelName:generateContent?key=$apiKey"
        val systemPrompt = buildSystemPrompt(allowSpoilers, userData, mediaFocus)
        val supportsTools = !modelName.startsWith("gemma", ignoreCase = true)

        val contentsList = mutableListOf<JsonObject>()
        val relevantHistory = conversationHistory.takeLast(12)
        for (msg in relevantHistory) {
            if (msg.isError || msg.text.isBlank()) continue
            contentsList.add(buildJsonObject {
                put("role", if (msg.isUser) "user" else "model")
                putJsonArray("parts") {
                    add(buildJsonObject { put("text", msg.text) })
                }
            })
        }
        contentsList.add(buildJsonObject {
            put("role", "user")
            putJsonArray("parts") {
                add(buildJsonObject { put("text", latestUserMessage) })
            }
        })

        fun buildToolsJson(): JsonArray {
            return buildJsonArray {
                if (supportsTools && userData.isNotEmpty()) {
                    add(buildJsonObject {
                        putJsonArray("functionDeclarations") {
                            add(buildJsonObject {
                                put("name", "getUserNotes")
                                put("description", "Look up the user's personal notes, thoughts, review, or rating for a specific anime or manga title in their AniList library.")
                                putJsonObject("parameters") {
                                    put("type", "OBJECT")
                                    putJsonObject("properties") {
                                        putJsonObject("title") {
                                            put("type", "STRING")
                                            put("description", "The title of the anime or manga to retrieve notes for.")
                                        }
                                    }
                                    putJsonArray("required") {
                                        add("title")
                                    }
                                }
                            })
                        }
                    })
                }
                if (useWebSearch && supportsTools) {
                    add(buildJsonObject {
                        putJsonObject("googleSearch") {}
                    })
                }
            }
        }

        val initialTools = buildToolsJson()

        val requestJson = buildJsonObject {
            putJsonObject("systemInstruction") {
                putJsonArray("parts") {
                    add(buildJsonObject { put("text", systemPrompt) })
                }
            }
            putJsonArray("contents") {
                contentsList.forEach { add(it) }
            }
            if (initialTools.isNotEmpty()) {
                put("tools", initialTools)
            }
        }

        val mediaType = "application/json; charset=utf-8".toMediaType()
        val requestBody = requestJson.toString().toRequestBody(mediaType)

        val request = Request.Builder()
            .url(endpoint)
            .post(requestBody)
            .build()

        var response = client.newCall(request).execute()
        var responseBody = response.body?.string() ?: throw IOException("Empty response from Gemini API")

        // Fallback retry without tools if custom model rejected tool definitions
        if (!response.isSuccessful && initialTools.isNotEmpty()) {
            val fallbackJson = buildJsonObject {
                putJsonObject("systemInstruction") {
                    putJsonArray("parts") {
                        add(buildJsonObject { put("text", systemPrompt) })
                    }
                }
                putJsonArray("contents") {
                    contentsList.forEach { add(it) }
                }
            }
            val fallbackBody = fallbackJson.toString().toRequestBody(mediaType)
            val fallbackReq = Request.Builder().url(endpoint).post(fallbackBody).build()
            val fallbackResp = client.newCall(fallbackReq).execute()
            if (fallbackResp.isSuccessful) {
                response = fallbackResp
                responseBody = fallbackResp.body?.string() ?: responseBody
            }
        }

        if (!response.isSuccessful) {
            val errorMsg = runCatching {
                val element = json.parseToJsonElement(responseBody).jsonObject
                element["error"]?.jsonObject?.get("message")?.jsonPrimitive?.contentOrNull
            }.getOrNull() ?: "Gemini API error (HTTP ${response.code})"
            throw IOException(errorMsg)
        }

        // Check if model returned a functionCall for getUserNotes
        val root = json.parseToJsonElement(responseBody).jsonObject
        val candidates = root["candidates"]?.jsonArray
        val firstCandidate = candidates?.firstOrNull()?.jsonObject
        val content = firstCandidate?.get("content")?.jsonObject
        val parts = content?.get("parts")?.jsonArray

        val functionCallPart = parts?.firstOrNull {
            it.jsonObject.containsKey("functionCall")
        }?.jsonObject?.get("functionCall")?.jsonObject

        if (functionCallPart != null) {
            val funcName = functionCallPart["name"]?.jsonPrimitive?.contentOrNull
            if (funcName == "getUserNotes") {
                val targetTitle = functionCallPart["args"]?.jsonObject?.get("title")?.jsonPrimitive?.contentOrNull ?: ""
                val cleanQuery = targetTitle.trim().lowercase()

                val matchedEntry = userData.firstOrNull {
                    it.titleUserPreferred.lowercase().contains(cleanQuery) ||
                    it.titleRomaji?.lowercase()?.contains(cleanQuery) == true ||
                    it.titleEnglish?.lowercase()?.contains(cleanQuery) == true ||
                    it.titleNative?.lowercase()?.contains(cleanQuery) == true
                }

                val noteResult = if (matchedEntry != null) {
                    if (!matchedEntry.notes.isNullOrBlank()) {
                        "Personal notes for '${matchedEntry.titleUserPreferred}': \"${matchedEntry.notes}\" (Status: ${matchedEntry.status}, Score: ${matchedEntry.score ?: "Unrated"})"
                    } else {
                        "User has '${matchedEntry.titleUserPreferred}' tracked (${matchedEntry.status}, Progress: ${matchedEntry.progress}, Score: ${matchedEntry.score ?: "Unrated"}), but has not written any personal notes."
                    }
                } else {
                    "No entry matching '$targetTitle' was found in user's library."
                }

                // Send function response back to Gemini to complete answer
                val followUpRequestJson = buildJsonObject {
                    putJsonObject("systemInstruction") {
                        putJsonArray("parts") {
                            add(buildJsonObject { put("text", systemPrompt) })
                        }
                    }
                    putJsonArray("contents") {
                        contentsList.forEach { add(it) }
                        // Model turn with functionCall
                        add(buildJsonObject {
                            put("role", "model")
                            putJsonArray("parts") {
                                add(buildJsonObject {
                                    putJsonObject("functionCall") {
                                        put("name", "getUserNotes")
                                        putJsonObject("args") {
                                            put("title", targetTitle)
                                        }
                                    }
                                })
                            }
                        })
                        // Tool response turn
                        add(buildJsonObject {
                            put("role", "function")
                            putJsonArray("parts") {
                                add(buildJsonObject {
                                    putJsonObject("functionResponse") {
                                        put("name", "getUserNotes")
                                        putJsonObject("response") {
                                            put("name", "getUserNotes")
                                            putJsonObject("content") {
                                                put("result", noteResult)
                                            }
                                        }
                                    }
                                })
                            }
                        })
                    }
                }

                val followUpBody = followUpRequestJson.toString().toRequestBody(mediaType)
                val followUpReq = Request.Builder().url(endpoint).post(followUpBody).build()
                val followUpResp = client.newCall(followUpReq).execute()
                val followUpBodyStr = followUpResp.body?.string() ?: throw IOException("Empty response from tool resolution")

                if (followUpResp.isSuccessful) {
                    return@withContext parseGeminiResponse(followUpBodyStr)
                }
            }
        }

        parseGeminiResponse(responseBody)
    }

    private fun parseGeminiResponse(responseBody: String): GenerateResult {
        val root = json.parseToJsonElement(responseBody).jsonObject
        val candidates = root["candidates"]?.jsonArray
        if (candidates.isNullOrEmpty()) {
            throw IOException("No response generated by Gemini model.")
        }

        val firstCandidate = candidates[0].jsonObject
        val content = firstCandidate["content"]?.jsonObject
        val parts = content?.get("parts")?.jsonArray

        val textBuilder = StringBuilder()
        val thinkingBuilder = StringBuilder()

        parts?.forEach { part ->
            val obj = part.jsonObject
            val partText = obj["text"]?.jsonPrimitive?.contentOrNull ?: ""
            val isThought = obj["thought"]?.jsonPrimitive?.booleanOrNull == true
            if (isThought) {
                thinkingBuilder.append(partText)
            } else {
                textBuilder.append(partText)
            }
        }

        var finalThinking = thinkingBuilder.toString().trim().ifBlank { null }
        var finalText = textBuilder.toString()

        // Also check for <thought>...</thought> or <think>...</think> XML tags in text
        if (finalThinking == null) {
            val thoughtRegex = Regex("""(?s)<(thought|think)>(.*?)</\1>""")
            val match = thoughtRegex.find(finalText)
            if (match != null) {
                finalThinking = match.groupValues[2].trim()
                finalText = finalText.replace(thoughtRegex, "").trim()
            }
        }

        val sources = mutableListOf<AiGroundingSource>()
        val groundingMetadata = firstCandidate["groundingMetadata"]?.jsonObject
        val groundingChunks = groundingMetadata?.get("groundingChunks")?.jsonArray

        groundingChunks?.forEach { chunk ->
            val web = chunk.jsonObject["web"]?.jsonObject
            val title = web?.get("title")?.jsonPrimitive?.contentOrNull
            val uri = web?.get("uri")?.jsonPrimitive?.contentOrNull
            if (!uri.isNullOrBlank()) {
                sources.add(AiGroundingSource(title = title ?: uri, url = uri))
            }
        }

        return GenerateResult(
            text = finalText.ifBlank { "I was unable to generate a response." },
            sources = sources.distinctBy { it.url },
            thinkingProcess = finalThinking
        )
    }

    private fun buildSystemPrompt(
        allowSpoilers: Boolean,
        userData: List<AiUserDataEntry>,
        mediaFocus: AiMediaFocusContext?
    ): String {
        val sb = StringBuilder()
        sb.appendLine("You are the AniSync AI Assistant, a knowledgeable, passionate, and helpful anime & manga companion inside the AniSync Android app.")
        sb.appendLine("You assist users with recommendations, character breakdowns, lore discussions, plot analysis, airing schedules, and reviewing their anime/manga lists.")
        sb.appendLine("You have broad encyclopedic knowledge of AniList anime/manga stats, characters, staff, and release details.")
        sb.appendLine()

        if (mediaFocus != null) {
            sb.appendLine("### CURRENTLY FOCUSED ANIME / MANGA CONTEXT (Opened from details page):")
            sb.appendLine("Title: ${mediaFocus.title}")
            mediaFocus.format?.let { sb.appendLine("Format: $it") }
            mediaFocus.status?.let { sb.appendLine("Status: $it") }
            mediaFocus.averageScore?.let { sb.appendLine("Average AniList Score: $it/100") }
            mediaFocus.episodes?.let { sb.appendLine("Episodes: $it") }
            if (mediaFocus.genres.isNotEmpty()) {
                sb.appendLine("Genres: ${mediaFocus.genres.joinToString(", ")}")
            }
            mediaFocus.studio?.let { sb.appendLine("Studio / Producers: $it") }
            mediaFocus.description?.let {
                sb.appendLine("Synopsis: ${it.take(1200)}")
            }
            if (mediaFocus.userStatus != null || !mediaFocus.userNotes.isNullOrBlank() || mediaFocus.userScore != null) {
                sb.appendLine("--- User's Personal Record on this Title ---")
                mediaFocus.userStatus?.let { sb.appendLine("User Status: $it") }
                mediaFocus.userProgress?.let { prog ->
                    val total = mediaFocus.userTotal?.let { "/$it" } ?: ""
                    sb.appendLine("User Progress: $prog$total")
                }
                mediaFocus.userScore?.let { sb.appendLine("User Score: $it/100") }
                if (!mediaFocus.userNotes.isNullOrBlank()) {
                    sb.appendLine("User's Personal Notes: \"${mediaFocus.userNotes}\"")
                }
            }
            sb.appendLine("Prioritize this specific title in your answers when relevant.")
            sb.appendLine()
        }

        if (allowSpoilers) {
            sb.appendLine("### SPOILER POLICY: SPOILERS ALLOWED")
            sb.appendLine("The user has enabled spoilers. You may discuss plot twists, manga source material beyond the anime, and ending details freely.")
        } else {
            sb.appendLine("### SPOILER POLICY: STRICTLY FORBIDDEN (ZERO SPOILERS)")
            sb.appendLine("The user has DISABLED spoilers. It is CRITICAL that you NEVER reveal any plot twists, climax events, character deaths, traitor reveals, identity reveals, or source material progression.")
            sb.appendLine("If the user asks 'What happens next?' or asks for spoilers, explicitly refuse and explain that Spoilers are turned OFF in the top bar. Keep all summaries focused exclusively on the premise, genres, production, and themes.")
        }
        sb.appendLine()

        if (userData.isNotEmpty()) {
            sb.appendLine("### USER'S PERSONAL ANILIST LIBRARY DATA (User Data toggle is ON):")
            sb.appendLine("You have access to a compact index of the user's anime and manga entries (status, score, and progress).")
            sb.appendLine("If the user asks about their specific personal notes, comments, or thoughts for a title, call the `getUserNotes` tool function to retrieve them.")
            for (entry in userData) {
                val titlePart = buildString {
                    append(entry.titleUserPreferred)
                    val altTitles = listOfNotNull(entry.titleRomaji, entry.titleEnglish)
                        .filter { it != entry.titleUserPreferred }
                        .distinct()
                    if (altTitles.isNotEmpty()) {
                        append(" (aka ${altTitles.joinToString(" / ")})")
                    }
                }
                val scorePart = entry.score?.let { "$it/100" } ?: "Unrated"
                val totalPart = entry.totalEpisodesOrChapters?.let { "/$it" } ?: ""
                val noteTag = if (!entry.notes.isNullOrBlank()) " [Has Notes]" else ""
                sb.appendLine("• [$titlePart] Type: ${entry.mediaType} | Status: ${entry.status} | Progress: ${entry.progress}$totalPart | Score: $scorePart$noteTag")
            }
            sb.appendLine()
        } else {
            sb.appendLine("### USER DATA POLICY: OFF")
            sb.appendLine("The user data toggle is currently OFF. You do not have access to their personal notes or private library records, but you can freely discuss general AniList facts, synopsis, and public stats.")
            sb.appendLine()
        }

        sb.appendLine("Format your response clearly using clean Markdown (bolding, bullet points, headers, quotes) formatted for mobile screens. Do not use random emoji spam.")
        return sb.toString()
    }

    suspend fun fetchNewsRadar(
        apiKey: String,
        modelName: String = "gemini-2.5-flash",
        topic: String = "All",
        currentDateTime: String = ""
    ): List<com.anisync.android.domain.ai.AiNewsItem> = withContext(Dispatchers.IO) {
        if (apiKey.isBlank()) {
            throw IllegalArgumentException("Please configure your Gemini API key in Settings > Gemini AI Assistant.")
        }

        val prompt = buildString {
            if (currentDateTime.isNotBlank()) {
                appendLine("CURRENT REAL-WORLD DATE & TIME: $currentDateTime.")
            }
            appendLine("You are the AI Anime News Radar for AniSync.")
            appendLine("Use Google Search Grounding to find breaking anime news, trailer drops, release dates, voice actor / studio announcements, and major industry updates from today or the past 48 hours relative to $currentDateTime.")
            if (topic != "All") {
                appendLine("Focus specifically on topic: $topic.")
            }
            appendLine("Format your response as a list of 5-8 news items formatted strictly as JSON array with this schema:")
            appendLine("""[{"title": "Headline without emojis", "summary": "2-3 sentence clear summary of the news and what fans should know", "category": "TRAILER|RELEASE|ANNOUNCEMENT|INDUSTRY", "timeAgo": "e.g. 2h ago or Today"}]""")
            appendLine("DO NOT include emojis in title, summary, or category. Output ONLY the valid JSON array and nothing else.")
        }

        val effectiveModel = modelName.ifBlank { "gemini-2.5-flash" }
        val url = "https://generativelanguage.googleapis.com/v1beta/models/$effectiveModel:generateContent?key=$apiKey"

        val requestJson = buildJsonObject {
            putJsonArray("contents") {
                add(buildJsonObject {
                    put("role", "user")
                    putJsonArray("parts") {
                        add(buildJsonObject { put("text", prompt) })
                    }
                })
            }
            putJsonArray("tools") {
                add(buildJsonObject {
                    putJsonObject("googleSearch") {}
                })
            }
        }

        val requestBody = requestJson.toString().toRequestBody("application/json; charset=utf-8".toMediaType())
        val request = Request.Builder().url(url).post(requestBody).build()

        val response = client.newCall(request).execute()
        val bodyString = response.body?.string() ?: throw IOException("Empty response from Gemini API")

        if (!response.isSuccessful) {
            val errorMsg = runCatching {
                val jsonEl = json.parseToJsonElement(bodyString).jsonObject
                jsonEl["error"]?.jsonObject?.get("message")?.jsonPrimitive?.contentOrNull
            }.getOrNull() ?: "API Error (${response.code})"
            throw IOException(errorMsg)
        }

        val root = json.parseToJsonElement(bodyString).jsonObject
        val candidate = root["candidates"]?.jsonArray?.firstOrNull()?.jsonObject
        val rawText = candidate?.get("content")?.jsonObject?.get("parts")?.jsonArray?.firstOrNull()
            ?.jsonObject?.get("text")?.jsonPrimitive?.contentOrNull.orEmpty()

        val sources = mutableListOf<AiGroundingSource>()
        val groundingMetadata = candidate?.get("groundingMetadata")?.jsonObject
        val searchChunks = groundingMetadata?.get("groundingChunks")?.jsonArray
        searchChunks?.forEach { chunkEl ->
            val web = chunkEl.jsonObject["web"]?.jsonObject
            val webTitle = web?.get("title")?.jsonPrimitive?.contentOrNull
            val webUri = web?.get("uri")?.jsonPrimitive?.contentOrNull
            if (!webUri.isNullOrBlank()) {
                sources.add(AiGroundingSource(title = webTitle ?: "Source", url = webUri))
            }
        }

        val jsonStartIndex = rawText.indexOf('[')
        val jsonEndIndex = rawText.lastIndexOf(']')
        if (jsonStartIndex != -1 && jsonEndIndex != -1 && jsonEndIndex > jsonStartIndex) {
            val jsonArrayStr = rawText.substring(jsonStartIndex, jsonEndIndex + 1)
            val parsed = runCatching {
                val array = json.parseToJsonElement(jsonArrayStr).jsonArray
                array.map { el ->
                    val obj = el.jsonObject
                    com.anisync.android.domain.ai.AiNewsItem(
                        title = obj["title"]?.jsonPrimitive?.contentOrNull?.replace(Regex("[\\p{So}\\p{Cn}]"), "")?.trim() ?: "Anime Update",
                        summary = obj["summary"]?.jsonPrimitive?.contentOrNull?.replace(Regex("[\\p{So}\\p{Cn}]"), "")?.trim() ?: "",
                        category = obj["category"]?.jsonPrimitive?.contentOrNull?.replace(Regex("[\\p{So}\\p{Cn}]"), "")?.trim() ?: "NEWS",
                        timeAgo = obj["timeAgo"]?.jsonPrimitive?.contentOrNull?.replace(Regex("[\\p{So}\\p{Cn}]"), "")?.trim() ?: "Recent",
                        sources = sources
                    )
                }
            }.getOrDefault(emptyList())
            if (parsed.isNotEmpty()) return@withContext parsed
        }

        listOf(
            com.anisync.android.domain.ai.AiNewsItem(
                title = "Latest Anime Radar",
                summary = rawText.take(500),
                category = "NEWS",
                timeAgo = "Recent",
                sources = sources
            )
        )
    }
}
