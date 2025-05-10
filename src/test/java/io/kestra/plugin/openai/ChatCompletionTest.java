package io.kestra.plugin.openai;

import com.theokanning.openai.completion.chat.ChatFunctionCall;
import com.theokanning.openai.completion.chat.ChatMessage;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Disabled;
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

        List<ChatMessage> messages = List.of(
            buildMessage("user","what is the capital of France?")
        );

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(Property.of(getApiKey()))
            .model(Property.of("gpt-4o"))
            .clientTimeout(30)
            .messages(Property.of(messages))
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);

        assertThat(runOutput.getChoices().getFirst().getMessage().getContent(), containsString("Paris"));
        assertThat(runOutput.getModel(), containsString("gpt-4o"));
        assertThat(runOutput.getUsage().getPromptTokens(), is(14L));
    }

    @Test
    void runPrompt() throws Exception {
        RunContext runContext = runContextFactory.of();

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(Property.of(getApiKey()))
            .model(Property.of("gpt-4o"))
            .prompt(Property.of("what is the capital of France?"))
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);

        assertThat(runOutput.getChoices().getFirst().getMessage().getContent(), containsString("Paris"));
        assertThat(runOutput.getModel(), containsString("gpt-4o"));
        assertThat(runOutput.getUsage().getPromptTokens(), is(14L));
    }

    @Test
    void runMessagesWithPrompt() throws Exception {
        RunContext runContext = runContextFactory.of();

        List<ChatMessage> messages = List.of(
            buildMessage("user","what is the capital of France?"),
            buildMessage("assistant","The capital of France is Paris.")
        );

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(Property.of(getApiKey()))
            .model(Property.of("gpt-4o"))
            .messages(Property.of(messages))
            .prompt(Property.of("and the capital of germany?"))
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);

        assertThat(runOutput.getChoices().getFirst().getMessage().getContent(), containsString("Berlin"));
        assertThat(runOutput.getModel(), containsString("gpt-4o"));
        assertThat(runOutput.getUsage().getPromptTokens(), is(35L));
    }

    @Test
    void runFunction() throws Exception {
        RunContext runContext = runContextFactory.of();

        List<ChatCompletion.PluginChatFunctionParameter> parameters = List.of(
            ChatCompletion.PluginChatFunctionParameter.builder()
                .name(Property.of("location"))
                .type(Property.of("string"))
                .description(Property.of("The city and state/province, and country, e.g. San Francisco, CA, USA"))
                .required(Property.of(Boolean.TRUE))
                .build(),
            ChatCompletion.PluginChatFunctionParameter.builder()
                .name(Property.of("unit"))
                .type(Property.of("string"))
                .description(Property.of("The temperature unit this city uses."))
                .required(Property.of(Boolean.TRUE))
                .build()
        );

        List<ChatCompletion.PluginChatFunction> functions = List.of(
            ChatCompletion.PluginChatFunction.builder()
                .name(Property.of("test"))
                .description(Property.of("finds the most relevant city and its temperature unit"))
                .parameters(Property.of(parameters))
                .build()

        );

        List<ChatMessage> messages = List.of(
            buildMessage("user","I was travelling along the Mediterranean coast, and I ended up in Lyon.")
        );

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(Property.of(getApiKey()))
            .model(Property.of("gpt-4o"))
            .messages(Property.of(messages))
            .functions(Property.of(functions))
            .functionCall(Property.of("auto"))
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);
        ChatFunctionCall functionCall = runOutput.getChoices().getFirst().getMessage().getFunctionCall();

        assertThat(functionCall.getName(), containsString("test"));
        assertThat(functionCall.getArguments().get("location").toString(), containsString("Lyon"));
        assertThat(runOutput.getModel(), containsString("gpt-4o"));
    }

    @Test
    void runFunctionWithEnumValues() throws Exception {
        RunContext runContext = runContextFactory.of();

        List<ChatCompletion.PluginChatFunctionParameter> parameters = List.of(
            ChatCompletion.PluginChatFunctionParameter.builder()
                .name(Property.of("rating"))
                .type(Property.of("string"))
                .description(Property.of("A rating of what the customer thought of our restaurant based on the review they wrote."))
                .required(Property.of(Boolean.TRUE))
                .enumValues(Property.of(List.of("excellent", "acceptable", "terrible")))
                .build(),
            ChatCompletion.PluginChatFunctionParameter.builder()
                .name(Property.of("food_eaten"))
                .type(Property.of("string"))
                .description(Property.of("A list of the food the customer ate, or 'Unknown' if they did not specify."))
                .required(Property.of(Boolean.TRUE))
                .build(),
            ChatCompletion.PluginChatFunctionParameter.builder()
                .name(Property.of("customer_name"))
                .type(Property.of("string"))
                .description(Property.of("The customer's name."))
                .required(Property.of(Boolean.TRUE))
                .build()
        );

        List<ChatCompletion.PluginChatFunction> functions = List.of(
            ChatCompletion.PluginChatFunction.builder()
                .name(Property.of("record_customer_rating"))
                .description(Property.of("Saves a customer's rating of our restaurant based on what they wrote in an online review."))
                .parameters(Property.of(parameters))
                .build()
        );

        List<ChatMessage> messages = List.of(
            buildMessage("user","My name is John Smith. I ate at your restaurant last week and order the steak. It was the worst steak I've even eaten. I will not be returning!")
        );

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(Property.of(getApiKey()))
            .model(Property.of("gpt-4o"))
            .messages(Property.of(messages))
            .functions(Property.of(functions))
            .functionCall(Property.of("record_customer_rating"))
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);
        ChatFunctionCall functionCall = runOutput.getChoices().getFirst().getMessage().getFunctionCall();

        assertThat(functionCall.getName(), containsString("record_customer_rating"));
        assertThat(functionCall.getArguments().get("rating").toString(), containsString("terrible"));
        assertThat(functionCall.getArguments()
            .get("food_eaten").toString()
            .toLowerCase(), containsString("steak"));
        assertThat(functionCall.getArguments().get("customer_name").toString(), containsString("John Smith"));
        assertThat(runOutput.getModel(), containsString("gpt-4o"));
    }

    private ChatMessage buildMessage(String role, String content) {
        ChatMessage message = new ChatMessage();
        message.setRole(role);
        message.setContent(content);

        return message;
    }
}
