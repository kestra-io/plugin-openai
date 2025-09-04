package io.kestra.plugin.openai;

import com.openai.models.chat.completions.ChatCompletionMessageFunctionToolCall;
import com.openai.models.chat.completions.ChatCompletionMessageToolCall;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledIf;

import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

@KestraTest
@DisabledIf(
    value = "canNotBeEnabled",
    disabledReason = "Needs an OpenAI API Key to work"
)
class ChatCompletionTest extends AbstractOpenAITest {
    @Inject
    private RunContextFactory runContextFactory;

    @Test
    void runMessages() throws Exception {
        RunContext runContext = runContextFactory.of();

        List<ChatCompletion.ChatMessage> messages = List.of(
            buildMessage("user", "what is the capital of France?")
        );

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(Property.ofValue(getApiKey()))
            .model(Property.ofValue("gpt-4o"))
            .clientTimeout(30)
            .messages(Property.ofValue(messages))
            .maxTokens(Property.ofValue(15L))
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);

        assertThat(runOutput.getChoices().getFirst().message().content().get(), containsString("Paris"));
        assertThat(runOutput.getModel(), containsString("gpt-4o"));
        assertThat(runOutput.getUsage().promptTokens(), is(14L));
    }

    @Test
    void runPrompt() throws Exception {
        RunContext runContext = runContextFactory.of();

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(Property.ofValue(getApiKey()))
            .model(Property.ofValue("gpt-4o"))
            .prompt(Property.ofValue("what is the capital of France?"))
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);

        assertThat(runOutput.getChoices().getFirst().message().content().get(), containsString("Paris"));
        assertThat(runOutput.getModel(), containsString("gpt-4o"));
        assertThat(runOutput.getUsage().promptTokens(), is(14L));
    }

    @Test
    void runMessagesWithPrompt() throws Exception {
        RunContext runContext = runContextFactory.of();

        List<ChatCompletion.ChatMessage> messages = List.of(
            buildMessage("user", "what is the capital of France?"),
            buildMessage("assistant", "The capital of France is Paris.")
        );

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(Property.ofValue(getApiKey()))
            .model(Property.ofValue("gpt-4o"))
            .messages(Property.ofValue(messages))
            .prompt(Property.ofValue("and the capital of germany?"))
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);

        assertThat(runOutput.getChoices().getFirst().message().content().get(), containsString("Berlin"));
        assertThat(runOutput.getModel(), containsString("gpt-4o"));
        assertThat(runOutput.getUsage().promptTokens(), is(35L));
    }

    @Test
    void runFunction() throws Exception {
        RunContext runContext = runContextFactory.of();

        List<ChatCompletion.PluginChatFunctionParameter> parameters = List.of(
            ChatCompletion.PluginChatFunctionParameter.builder()
                .name(Property.ofValue("location"))
                .type(Property.ofValue("string"))
                .description(Property.ofValue("The city and state/province, and country, e.g. San Francisco, CA, USA"))
                .required(Property.ofValue(Boolean.TRUE))
                .build(),
            ChatCompletion.PluginChatFunctionParameter.builder()
                .name(Property.ofValue("unit"))
                .type(Property.ofValue("string"))
                .description(Property.ofValue("The temperature unit this city uses."))
                .required(Property.ofValue(Boolean.TRUE))
                .build()
        );

        List<ChatCompletion.PluginChatFunction> functions = List.of(
            ChatCompletion.PluginChatFunction.builder()
                .name(Property.ofValue("test"))
                .description(Property.ofValue("finds the most relevant city and its temperature unit"))
                .parameters(Property.ofValue(parameters))
                .build()
        );

        List<ChatCompletion.ChatMessage> messages = List.of(
            buildMessage("user", "I was travelling along the Mediterranean coast, and I ended up in Lyon.")
        );

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(Property.ofValue(getApiKey()))
            .model(Property.ofValue("gpt-4o"))
            .messages(Property.ofValue(messages))
            .functions(Property.ofValue(functions))
            .functionCall(Property.ofValue("test"))
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);
        var functionCall = runOutput.getChoices().getFirst().message().toolCalls()
            .get().getFirst().asFunction().function();

        assertThat(functionCall.name(), containsString("test"));
        assertThat(functionCall._arguments().asString().orElse(""), containsString("Lyon"));
        assertThat(runOutput.getModel(), containsString("gpt-4o"));
    }

    @Test
    void runFunctionWithEnumValues() throws Exception {
        RunContext runContext = runContextFactory.of();

        List<ChatCompletion.PluginChatFunctionParameter> parameters = List.of(
            ChatCompletion.PluginChatFunctionParameter.builder()
                .name(Property.ofValue("rating"))
                .type(Property.ofValue("string"))
                .description(Property.ofValue("A rating of what the customer thought of our restaurant based on the review they wrote."))
                .required(Property.ofValue(Boolean.TRUE))
                .enumValues(Property.ofValue(List.of("excellent", "acceptable", "terrible")))
                .build(),
            ChatCompletion.PluginChatFunctionParameter.builder()
                .name(Property.ofValue("food_eaten"))
                .type(Property.ofValue("string"))
                .description(Property.ofValue("A list of the food the customer ate, or 'Unknown' if they did not specify."))
                .required(Property.ofValue(Boolean.TRUE))
                .build(),
            ChatCompletion.PluginChatFunctionParameter.builder()
                .name(Property.ofValue("customer_name"))
                .type(Property.ofValue("string"))
                .description(Property.ofValue("The customer's name."))
                .required(Property.ofValue(Boolean.TRUE))
                .build()
        );

        List<ChatCompletion.PluginChatFunction> functions = List.of(
            ChatCompletion.PluginChatFunction.builder()
                .name(Property.ofValue("record_customer_rating"))
                .description(Property.ofValue("Saves a customer's rating of our restaurant based on what they wrote in an online review."))
                .parameters(Property.ofValue(parameters))
                .build()
        );

        List<ChatCompletion.ChatMessage> messages = List.of(
            buildMessage("user", "My name is John Smith. I ate at your restaurant last week and order the steak. It was the worst steak I've even eaten. I will not be returning!")
        );

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(Property.ofValue(getApiKey()))
            .model(Property.ofValue("gpt-4o"))
            .messages(Property.ofValue(messages))
            .functions(Property.ofValue(functions))
            .functionCall(Property.ofValue("auto"))
            .maxTokens(Property.ofValue(50L))
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);
        var functionCall = runOutput.getChoices().getFirst().message()
                .toolCalls().orElse(List.of())
                .getFirst().asFunction().function();

        assertThat(functionCall.name(), containsString("record_customer_rating"));
        assertThat(functionCall._arguments().asString().orElse(""), containsString("terrible"));
        assertThat(functionCall._arguments().asString().orElse(""), containsString("steak"));
        assertThat(functionCall._arguments().asString().orElse(""), containsString("John Smith"));
        assertThat(runOutput.getModel(), containsString("gpt-4o"));
    }

    private ChatCompletion.ChatMessage buildMessage(String role, String content) {
        return ChatCompletion.ChatMessage.builder().role(role).content(content).build();
    }
}
