"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports._chatCompletionMessagesToString = exports._isTrajectorySuperset = void 0;
function _normalizeToolCall(toolCall) {
    if ("function" in toolCall &&
        toolCall.function != null &&
        typeof toolCall.function === "object") {
        return {
            name: toolCall.function.name,
            args: toolCall.function.arguments,
        };
    }
    return toolCall;
}
function _extractToolCalls(messages) {
    const toolCalls = [];
    for (const message of messages) {
        if (message.tool_calls) {
            toolCalls.push(...message.tool_calls.map(_normalizeToolCall));
        }
    }
    return toolCalls;
}
function _isTrajectorySuperset(outputs, referenceOutputs) {
    const outputToolCalls = _extractToolCalls(outputs);
    const referenceToolCalls = _extractToolCalls(referenceOutputs);
    const outputToolCounts = new Map();
    const referenceToolCounts = new Map();
    for (const call of outputToolCalls) {
        outputToolCounts.set(call.name, (outputToolCounts.get(call.name) ?? 0) + 1);
    }
    for (const call of referenceToolCalls) {
        referenceToolCounts.set(call.name, (referenceToolCounts.get(call.name) ?? 0) + 1);
    }
    const allTools = new Set([
        ...outputToolCounts.keys(),
        ...referenceToolCounts.keys(),
    ]);
    for (const name of allTools) {
        if ((outputToolCounts.get(name) ?? 0) < (referenceToolCounts.get(name) ?? 0)) {
            return false;
        }
    }
    return true;
}
exports._isTrajectorySuperset = _isTrajectorySuperset;
function _chatCompletionMessagesToString(messages) {
    function formatMessage(message) {
        let content = message.content ?? "";
        // Handle tool/function calls
        if (message.tool_calls) {
            const toolCallsStr = message.tool_calls
                .map((call) => {
                const func = call.function ?? {};
                return `<tool_call>\n<name>${func.name ?? ""}</name>\n<arguments>${func.arguments ?? ""}</arguments>\n</tool_call>`;
            })
                .join("\n");
            content = content ? `${content}\n${toolCallsStr}` : toolCallsStr;
        }
        // Handle tool call results
        if (message.tool_call_id) {
            content = `<tool_result>\n<id>${message.tool_call_id}</id>\n<content>${content}</content>\n</tool_result>`;
        }
        return `<${message.role ?? ""}>\n${content}\n</${message.role ?? ""}>`;
    }
    return messages.map(formatMessage).join("\n\n");
}
exports._chatCompletionMessagesToString = _chatCompletionMessagesToString;
