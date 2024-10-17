package io.kestra.plugin.openai;

import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;


@KestraTest
@Disabled("Needs an OpenAI API Key to work")
class CreateImageTest {
    @Inject
    private RunContextFactory runContextFactory;

    private final String apiKey = "";


    @Test
    void runPromptUrl() throws Exception {
        RunContext runContext = runContextFactory.of();

        CreateImage task = CreateImage.builder()
            .apiKey(Property.of(this.apiKey))
            .prompt(Property.of("A funny cat in a black suit"))
            .size(Property.of(CreateImage.SIZE.SMALL))
            .build();

        CreateImage.Output runOutput = task.run(runContext);

        assertThat(runOutput.getImages().size(), is(1));
    }

    @Test
    void runPromptB64Json() throws Exception {
        RunContext runContext = runContextFactory.of();

        CreateImage task = CreateImage.builder()
            .apiKey(Property.of(this.apiKey))
            .prompt(Property.of("A funny cat in a black suit"))
            .size(Property.of(CreateImage.SIZE.SMALL))
            .download(Property.of(Boolean.TRUE))
            .build();

        CreateImage.Output runOutput = task.run(runContext);

        assertThat(runOutput.getImages().size(), is(1));
    }
}
