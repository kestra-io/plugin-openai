package io.kestra.plugin.openai;

import com.openai.client.OpenAIClient;
import com.openai.models.images.ImageGenerateParams;
import com.openai.models.images.ImagesResponse;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.runners.RunContext;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.experimental.SuperBuilder;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collection;
import java.util.List;
import java.util.UUID;

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
    private Property<SIZE> size = Property.ofValue(SIZE.LARGE);

    @Schema(
        title = "Whether to download the generated image",
        description = "If enable, the generated image will be downloaded inside Kestra's internal storage. Else, the URL of the generated image will be available as task output."
    )
    @Builder.Default
    @NotNull
    private Property<Boolean> download = Property.ofValue(Boolean.FALSE);

    @Override
    public CreateImage.Output run(RunContext runContext) throws Exception {
        OpenAIClient client = this.openAIClient(runContext);

        String user = runContext.render(this.user == null ? null : runContext.render(this.user).as(String.class).orElseThrow());
        String prompt = runContext.render(this.prompt).as(String.class).orElseThrow();
        ImageGenerateParams.ResponseFormat responseFormat = runContext.render(this.download).as(Boolean.class).orElseThrow() ?
                ImageGenerateParams.ResponseFormat.B64_JSON: ImageGenerateParams.ResponseFormat.URL;

       ImagesResponse imageResult = client.images().generate(ImageGenerateParams.builder()
            .prompt(prompt)
            .size(runContext.render(this.size).as(SIZE.class).orElseThrow().getSize())
            .n(this.n)
            .responseFormat(responseFormat)
            .user(user)
            .build());


        List<URI> files = new ArrayList<>();
        imageResult.data().stream()
            .flatMap(Collection::stream)
            .forEach(throwConsumer(image -> {
            if (runContext.render(this.download).as(Boolean.class).orElseThrow()) {
                if (image.b64Json().isPresent()) {
                    files.add(runContext.storage().putFile(this.downloadB64Json(image.b64Json().get())));
                }
            } else {
                image.url().ifPresent(url -> files.add(URI.create(url)));
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
        SMALL("256x256", ImageGenerateParams.Size._256X256),
        MEDIUM("512x512", ImageGenerateParams.Size._512X512),
        LARGE("1024x1024", ImageGenerateParams.Size._1024X1024);

        private final String value;
        private final ImageGenerateParams.Size size;

        SIZE(String value, ImageGenerateParams.Size size) {
            this.value = value;
            this.size = size;
        }

        public ImageGenerateParams.Size getSize() {
            return size;
        }

    }

}
