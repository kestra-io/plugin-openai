package io.kestra.plugin.openai;

import com.theokanning.openai.image.CreateImageRequest;
import com.theokanning.openai.image.ImageResult;
import com.theokanning.openai.service.OpenAiService;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.runners.RunContext;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.*;
import lombok.experimental.SuperBuilder;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.UUID;

import jakarta.validation.constraints.NotNull;

import static io.kestra.core.utils.Rethrow.throwConsumer;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Given a prompt, create an image with OpenAI.",
    description = "For more information, refer to the [OpenAI Image Generation API docs](https://platform.openai.com/docs/api-reference/images/create)."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            code = """
                id: openai
                namespace: company.team

                tasks:
                  - id: create_image
                    type: io.kestra.plugin.openai.CreateImage
                    prompt: A funny cat in a black suit
                    apiKey: <your-api-key>
                    download: true
                    n: 5
                """
        )
    }
)
public class CreateImage extends AbstractTask implements RunnableTask<CreateImage.Output> {
    @Schema(
        title = "Message to send to the API as prompt"
    )
    @NotNull
    private Property<String> prompt;

    @Schema(
        title = "The number of images to generate; must be between 1 and 10."
    )
    private Integer n;

    @Schema(
        title = "The size of the generated images."
    )
    @Builder.Default
    @NotNull
    private Property<SIZE> size = Property.of(SIZE.LARGE);

    @Schema(
        title = "Whether to download the generated image",
        description = "If enable, the generated image will be downloaded inside Kestra's internal storage. Else, the URL of the generated image will be available as task output."
    )
    @Builder.Default
    @NotNull
    private Property<Boolean> download = Property.of(Boolean.FALSE);

    @Override
    public CreateImage.Output run(RunContext runContext) throws Exception {
        OpenAiService client = this.client(runContext);

        String user = runContext.render(this.user == null ? null : runContext.render(this.user).as(String.class).orElseThrow());
        String prompt = runContext.render(this.prompt).as(String.class).orElseThrow();

        ImageResult imageResult = client.createImage(CreateImageRequest.builder()
            .prompt(prompt)
            .size(runContext.render(this.size).as(SIZE.class).orElseThrow().getSize())
            .n(this.n)
            .responseFormat(runContext.render(this.download).as(Boolean.class).orElseThrow() ? "b64_json" : "url")
            .user(user)
            .build()
        );

        List<URI> files = new ArrayList<>();
        imageResult.getData().forEach(throwConsumer(image -> {
            if (runContext.render(this.download).as(Boolean.class).orElseThrow()) {
                files.add(runContext.storage().putFile(this.downloadB64Json(image.getB64Json())));
            } else {
                files.add(URI.create(image.getUrl()));
            }
        }));

        return Output.builder().images(files).build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {

        @Schema(
            title = "Generated images"
        )
        private List<URI> images;
    }

    private File downloadB64Json(String encodedImage) throws IOException {
        File image = File.createTempFile("openai-" + UUID.randomUUID(), ".png");
        byte[] decodedBytes = Base64.getDecoder().decode(encodedImage);
        FileUtils.writeByteArrayToFile(image, decodedBytes);

        return image;
    }

    protected enum SIZE {
        SMALL("256x256"),
        MEDIUM("512x512"),
        LARGE("1024x1024");

        private final String value;

        SIZE(String value) {
            this.value = value;
        }

        public String getSize() {
            return value;
        }

    }

}
