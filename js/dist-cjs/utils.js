"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports._runEvaluator = exports.processScore = exports._normalizeToOpenAIMessagesList = exports._convertToOpenAIMessage = void 0;
const messages_1 = require("@langchain/core/messages");
const openai_1 = require("@langchain/openai");
const jestlike_1 = require("langsmith/utils/jestlike");
const _convertToOpenAIMessage = (message) => {
    if ((0, messages_1.isBaseMessage)(message)) {
        return (0, openai_1._convertMessagesToOpenAIParams)([message])[0];
    }
    else {
        return message;
    }
};
exports._convertToOpenAIMessage = _convertToOpenAIMessage;
const _normalizeToOpenAIMessagesList = (messages) => {
    if (!messages) {
        return [];
    }
    let messagesList;
    if (!Array.isArray(messages)) {
        if ("messages" in messages && Array.isArray(messages.messages)) {
            messagesList = messages.messages;
        }
        else {
            throw new Error(`If passing messages as an object, it must contain a "messages" key`);
        }
    }
    else {
        messagesList = messages;
    }
    return messagesList.map(exports._convertToOpenAIMessage);
};
exports._normalizeToOpenAIMessagesList = _normalizeToOpenAIMessagesList;
const processScore = (_, value) => {
    if (typeof value === "object") {
        if (value != null && "score" in value) {
            return [
                value.score,
                "reasoning" in value && typeof value.reasoning === "string"
                    ? value.reasoning
                    : undefined,
            ];
        }
        else {
            throw new Error(`Expected a dictionary with a "score" key, but got "${JSON.stringify(value, null, 2)}"`);
        }
    }
    return [value];
};
exports.processScore = processScore;
const _runEvaluator = async (runName, scorer, feedbackKey, extra) => {
    const runScorer = async (params) => {
        let score = await scorer(params);
        let reasoning;
        const results = [];
        if (!Array.isArray(score) && typeof score === "object") {
            for (const [key, value] of Object.entries(score)) {
                const [keyScore, reasoning] = (0, exports.processScore)(key, value);
                results.push({ key, score: keyScore, comment: reasoning });
            }
        }
        else {
            if (Array.isArray(score)) {
                reasoning = score[1];
                score = score[0];
            }
            results.push({ key: feedbackKey, score, comment: reasoning });
        }
        if (results.length === 1) {
            return results[0];
        }
        else {
            return results;
        }
    };
    if ((0, jestlike_1.isInTestContext)()) {
        const res = await (0, jestlike_1.wrapEvaluator)(runScorer)(extra ?? {}, {
            name: runName,
        });
        return res;
    }
    else {
        const res = await runScorer(extra ?? {});
        return res;
    }
};
exports._runEvaluator = _runEvaluator;
